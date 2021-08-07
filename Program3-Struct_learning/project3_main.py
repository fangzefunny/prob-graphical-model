'''
Project3: Structure learning with incomplete data 

'''

import os 
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from collections import namedtuple
from utils import fac_prod, fac_take, bayesNet, inference, normalize

'''
PART 0: Comment on imported package

The files in utils is well-developed stable functions and 
classes I wrote in the past few homeworks & projects. 

* fac_prod, fac_take,

  (find more detail in the definition in utils.prob_tools)

  factor: probabiility 
  fac_prod: multiplication over 2 probability 
  fac_sum: marginal over some variables

  Classes and functions that allow us directly translate
  the mathematical equation. Previously parts of the code are 
  from internet. In the new version, I wrote everthing myself.

  For example, to calculate \sum_B p(A|B)p(B):

  p_AB = factor( [ 'A', 'B'], np.array([ 2, 3]))
  p_B  = factor( ['B'], np.array([3]))
  P_Ab=2  = fac_take( fac_prod( p_AB, p_B), ['B'], 2)

* _sample and nomralization

  (find more detail in the definition in utils.prob_tools)

  _sample: sam ~ uni(0,1)
  normalize: normalize the distribution and ensure it sums to 1. 

* bayesNet: 

  (find more detail in the definition in utils.basic)

  Bayesian network module that allows quickly 
  
  1. manipulates nodes and links, 
  2. visualize the structure and store cpt parameters.
  3. assign and switch inference & learning algorithms. 

* inference

  (find more detial in the definition in utils.inference)

  algorithm to make inference. Currently only belief propagation
  is saved. Later, sampling methoda and variational inference will 
  also be added. 

'''

'''
PART1: Parameter learning algorithm

Count and find the parameter that fit 
'''

class counter:
    '''Parameter Learning
    Count and learn the parameter of a BN
    '''
    def __init__( self, graph, tol =1e-1, max_epi =20, verbose=False, infer_mode='EM'):
        # init same hyperparamters
        self.graph = graph
        self.tol = tol
        self.max_epi = max_epi
        self.verbose = verbose
        self.infer_mode = infer_mode

    def _empty_counts( self):
        self.par_config = []

    def _count_data( self, data, weights=[]):
        '''Count for each parent config 
        
        for a node that satisfies each parent configuration

        Input: 
            data: data we want to learn from 

        args:
            weights: this is created for infered data, whch has weights
                     for each datum. When we count, we need to consider 
                     the weight.
        
        Output:
            A dictionary that contains the number of data that satisfies
            node: xi = k and parent configurate: pi(xi) = j. 
        '''
        counts = {}
        for node_name in self.graph.name_nodes:
            # get node state: Xi = [ 0, 1, ..., K]
            node_idx = self.graph.name_nodes.index( node_name)
            i_node = self.graph.get_node( node_name)
            # get all parent configs for Xi: pi(Xi) = [ 0, 1, ..., J]
            par_config = i_node.par_configs
            count_mat = np.zeros( [ len(par_config), i_node.nstates])
            # start filtering data
            for row, config_j in enumerate(par_config):
                config_str = np.array(list(config_j))
                eq_idx = np.where( config_str == '=')[0]
                par_names = config_str[eq_idx-1]
                par_states = config_str[eq_idx+1]
                data_filter = 1.
                # if node has parents
                if len(par_names):
                    # Example: par_name: 'A', par_idx (col_idx in data): 0, par_state: '1'
                    for par_name, par_state in zip( par_names, par_states):
                        par_idx = self.graph.name_nodes.index( par_name)
                        # I( pi(Xi) = j)
                        data_filter *= ( data[ :, par_idx] == par_state)
                        # I( Xi = k)
                        for col, state in enumerate(i_node.states):
                            meet_requirements = data_filter * ( data[ :, node_idx] == state) 
                            if len(weights):
                                # if data is infered data 
                                count_mat[ row, col] = np.sum( meet_requirements * weights)
                            else:
                                count_mat[ row, col] = np.sum( meet_requirements)
                else:
                    # if node parents I(Xi = k)
                    for col, state in enumerate(i_node.states):
                        meet_requirements = data_filter * ( data[ :, node_idx] == state) 
                        if len(weights):
                            count_mat[ row, col] = np.sum( meet_requirements * weights)
                        else:
                            count_mat[ row, col] = np.sum( meet_requirements)
            # save counts to a dictionary 
            counts[ node_name] = count_mat
        return counts

    def _count_to_params( self, counts):
        '''Estimate parameters based on counts

        Input: 
            counts: A dictionary that contains the counts for each
                    node & its parent configuration. 

        Output: 
            params: p(theta|D), a dict of parameters learned from data. 
        '''
        params = dict()
        for node_name in self.graph.name_nodes:
            i_node = self.graph.get_node( node_name)
            par_config = i_node.par_configs
            i_params = dict()
            for row, config_j in enumerate( par_config):
                i_params[ config_j] = list(normalize( counts[node_name][ row, :]))
            params[ node_name] = i_params

        return params

    def best_params( self, data, weights =[]):
        '''Learn params from data    
        '''
        comp_counts = self._count_data( data, weights)
        params = self._count_to_params( comp_counts) 
        return params

'''
PART2: Local search algorithm

Hill climbing
'''

class hill_climbing:

    def __init__( self, graph, tol=1e-3, verbose=True, max_epi=200):
        self.graph = graph     # get the BN
        # some hyperparameters
        self.tol   = tol       # tolerance to decide convergence 
        self.verbose = verbose # visualize to keep track of convergence?
        self.max_epi = 20     # max iteration 
        # init some useful axuilary functions 
        self._init_cache()

    def clear_cache( self):
        self.BIC_cache = dict()

    def _init_cache( self):
        '''Save the BIC score for structure

        The idea of creating it is to reduce the computation complexity.
        Calculating the BIC score is the most computation demanding part.
        The logic is, if the structure is the same, the score of this
        structure should be the same. So we can reuse the previous computed 
        BIC score, params, loglike and structure for this structure without 
        calculating computing the loglikelihood
        
        Follow the idea of dynmaic programming
        '''
        self.BIC_cache = dict()
        self.explore_results = namedtuple( 'result', ('bic', 'logLike', 'struct', 'params'))

    def search_struct_and_params( self, data, weights):
        '''learn the best structure and parameters

        This is where I implment the pseudo-code (Algorithm1 in the report),
        It will return convergence conditions and just manipulate nodes and links in 
        the self.graph.

        Input:
            data: that we want to learn structure and parameters from
            weight: weights of the data

        Output: 
            best_struct: Given weighted data, the best possible structure
            best_params: Given weighted data, the best parameter
            best_bic: Given weighed data, the best 
            best_logLike: Given weighed data, the best loglikelihood 
        '''
        done = False
        self.deltas = []
        # init struct and param 
        best_struct = self.graph.print_struct()
        best_params = self.graph.print_params()
        best_bic, best_logLike = self.graph.cal_bic( data, weights)
        epi = 0
        # because the data has been changed, the memory is no longer useful
        self.clear_cache()
        # while not converge  
        while not done:
            #prepare for the new iteration
            old_bic = best_bic
            # save the current state
            self.explore_lst = [self.explore_results( bic=best_bic, 
                                                      logLike=best_logLike,
                                                      params=best_params, 
                                                      struct=best_struct)]
            # start exploring each node 
            for i_node_name in self.graph.name_nodes:
                self._explore_and_maxBIC(i_node_name, best_struct, data, weights)

            # get the best_bic, the best struct, and best params 
            bics = self.explore_results( *zip( *self.explore_lst)).bic
            best_idx = np.argmax( bics)
            best_struct = self.explore_lst[best_idx].struct
            best_params = self.explore_lst[best_idx].params
            best_bic    = self.explore_lst[best_idx].bic
            best_logLike = self.explore_lst[best_idx].logLike

            # check convergence 
            delta = abs(best_bic - old_bic)
            self.deltas.append( delta)
            if delta < self.tol:
                done = True 
            if self.verbose:
                print( 'At epi {}, the detla {:.4f}, the best struct:{}, best_bic:{}'
                                    .format( epi, delta, best_struct, best_bic))
            epi += 1 
            
        return best_struct, best_params, best_bic, best_logLike
        
    def _explore_and_maxBIC( self, i_node_name, curr_struct, data, weights):
        '''Explore and maxmize BIC to a node 

        Key implementation. According to the project instruction,
        the exploration to a node should follow: 
        * remove
        * change direction
        * add
        The general idea in this implementation is to manipulate
        the relationship between parents and childrens of node Xi. 
        Because For another Xj not= Xi, if there is a link between Xj
        and Xi, Xj is either its parent or its child.
        However, when explore, we only consider "parent --> i_node". Think
        about factorization of a BN, what we need is p(Xi| pi(Xi)).

        Input: 
            i_node_name: target node Xi, index by name 
            curr_struct: the best structure at the previous iteration,
                         this is the benchmark for the current iteration.
            data, weight: where we want to learn our structure and parameter from

        Output: 
            best_bic: highest BIC score 
            best_struct: structure the return the highest BIC
            best_params: Given structure, the best parameter that return
                         the highest BIC
        '''
        ## 1. Exploring the structure of the node
        
        # get the current structure 
        self.graph.load_struct( curr_struct)
        # get the node
        parent_lst = [ par_node.name for par_node in self.graph.get_node( i_node_name).parents]
        children_lst = [ child_node.name for child_node in self.graph.get_node( i_node_name).children]

        # remove edges (links)
        if len(parent_lst):
            for par_node_name in parent_lst: 
                self.graph.load_struct( curr_struct)
                self.explore_lst += self.explore( i_node_name, 
                                                   par_node_name, 
                                                   'remove', 
                                                   data,
                                                   weights)
                
        # reverse edges (links)
        if len(parent_lst):
            for par_node_name in parent_lst:  
                self.graph.load_struct( curr_struct)
                self.explore_lst += self.explore( i_node_name, 
                                                   par_node_name, 
                                                   'reverse', 
                                                   data,
                                                   weights)
                
        # add edges (links)
        # all possible Xj (not Xi, not pi(Xi), not child(Xi))
        other_nodes = self.graph.name_nodes.copy()
        other_nodes.pop( other_nodes.index(i_node_name))
        for j_node_name in other_nodes:
            if (j_node_name not in parent_lst) and (
                        j_node_name not in children_lst):
                self.graph.load_struct( curr_struct)
                self.explore_lst += self.explore( i_node_name,
                                                   j_node_name,
                                                   'add',
                                                   data,
                                                   weights)
                

    def explore( self, i_node_name, other_node_name, mode, data, weights):
        '''Explore the node structure

        In this function, we 
        * explore the local sturcture
        * learn parameter and computer bic
        * cache the learnt struct, params, bic

        Input:
            i_node: the target node, Xi
            other_node: other node Xj
            mode: either "remove edges", "reverse edges", "add edges"
            data: data where we want to learn from 
        
        Output:
            self.explore_lst: show the results of exploration 
        '''
        if mode == 'remove':
            self.graph.remove_edges([ other_node_name, i_node_name])
        elif mode == 'reverse':
            self.graph.reverse_edges([ other_node_name, i_node_name])
        elif mode == 'add':
            self.graph.add_edges( [i_node_name, other_node_name])
        else:
            raise Exception( 'choose the correct exploration')
        struct = self.graph.print_struct()
        key = str(struct)
        if key in self.BIC_cache.keys():
            bic = self.BIC_cache[ key]['bic']
            params = self.BIC_cache[ key]['params']
            logLike = self.BIC_cache[ key]['logLike']
        else:
            # learn the best params given struct 
            self.graph.load_struct(struct)
            if self.graph.isloopy():
                # do not learn parameter and give infinite 
                # low score to loopy structure
                params = None 
                bic = -np.inf 
                logLike = None
            else: 
                params = self.graph.best_params( data, weights)
                # calculate bic given params and struct
                self.graph.load_params(params)
                bic, logLike = self.graph.cal_bic( data, weights)
            # save to cache to avoid repeated computing the same
            content = dict()
            content['bic'] = bic
            content['params'] = params 
            content['struct'] = struct
            content['logLike'] = logLike
            self.BIC_cache[key] = content
        # show the explored structure and the bic  
        #print( 'struct: {}, bic: {}, logLike: {}'.format( str(struct), bic, logLike))
        # save the explore results
        return [self.explore_results( bic=bic, logLike=logLike, params=params, struct=struct)]

'''
PART3: Structure EM.
'''

class struct_EM:

    def __init__( self, graph, tol=1e-3, verbose=True, max_epi=200):
        self.graph = graph     # get the BN
        # some hyperparameters
        self.tol   = tol       # tolerance to decide convergence 
        self.verbose = verbose # visualize to keep track of convergence?
        self.max_epi = 10      # max iteration 
        self.infer_mode = 'EM'
        self.search_algs = hill_climbing( self.graph)
        # init some useful axuilary functions 
    
    def learn_structure_and_params( self, data):
        '''main function

        Where I conducted the pesudo code algorithm2.
        '''
        # init the BN structure, multiple way,
        # here we can use the exsited data
        self.graph.init_params( 'uniform')
        
        done = False
        epi = -1
        deltas = []
        bics   = []
        logLikes = []
        old_struct = self.graph.print_struct()
        old_params_lst = self.graph.all_params()
        
        while not done:
            if self.verbose:
                print( '=====> Start infer data ===>')
            # E-step, infer data 
            comp_data, incomp_data = self._split_data( data)
            comp_weights = []
            if incomp_data.shape[0] != 0:
                infer_data, weights  = self._infer_miss_data( incomp_data)
                comp_data, comp_weights = self._concat_data( comp_data, infer_data, weights)
            
            # M-step, G, theta = argmax Eq(BIC)
            new_struct, new_params, bic, logLike = self.search_algs.search_struct_and_params(
                                          comp_data, comp_weights)
            bics.append(bic)
            logLikes.append( logLike)
            self.graph.load_struct(new_struct)
            self.graph.load_params(new_params)
            new_params_lst = self.graph.all_params()
            # check convergence:
            if len(set(new_struct) - set(old_struct))==0:
                delta = np.sum( abs(new_params_lst - old_params_lst))
                deltas.append(delta)
                if  delta < self.tol:
                    done = True
            epi += 1
            if (epi > self.max_epi):
                done = True 

            if self.verbose:
                print( 'EMIteration:{}, the current struct:{}, the bic: {}, the logLike: {}'
                                    .format(epi, new_struct, bic, logLike))

            # prepare for the next iteration 
            old_struct = new_struct
            old_params_lst = new_params_lst

        return bics, logLikes        
        
    def _concat_data( self, comp_data, infer_data, weights):
        comp_weights = np.ones( [comp_data.shape[0],])
        new_data = np.concatenate( (comp_data, infer_data), axis=0)
        new_weights = np.concatenate( (comp_weights, weights))
        return new_data, new_weights

    def _split_data( self, data ):
            '''Identify complete and incomplete part of data
            '''
            comp_data = []
            incomp_data = []
            for datum in data:    
                if np.sum( datum == '-999'):    
                    incomp_data.append( datum)
                else:
                    comp_data.append( datum)
            return np.array(comp_data), np.array(incomp_data)

    def _infer_miss_data( self, incomp_data):
        '''Infer the missing data

        This is implementation of the E step in the EM method.
        Using inference algorithm we learned in the previous lecture.
        Here we are using belief propgation. An interesting discovery is
        that when the structure is simple, BP algorithm can also do well
        in the loopy Bayesian network.

        Note that if we are using EM to learn, the inference method we want
        to use is posterior of probability. If hard-EM, we are doing MPE.

        Input: 
            incomp_data: incomplete data 
        
        Output:
            complete data filled with inferred data,
            and the corresponding weights 
        '''
        infer_data = []
        infer_weights = []
        for t in range(incomp_data.shape[0]):
            # for each datum (data in a row)
            datum = incomp_data[t, :]
            # form evidence E1=e1. E2=e2(maybe I want to improve 
            # this step in the future ) because current BP input
            # is a str 
            evid = ''
            for idx, state in enumerate(datum):
                # if missing data, do not add to evidence 
                if state == '-999':
                    pass
                # add observed data to the evidence 
                else:
                    node_name = self.graph.name_nodes[ idx]
                    evid += '{}={},'.format( node_name, state)
            
            # clear the belief table 
            self.graph.infer.reset()
            # if EM use, sum-prodct, if hard-EM, MPE, max-product
            if self.infer_mode == 'EM':
                belief = self.graph.infer.inference( evidence = evid, 
                                            mode = 'sum-product')
            elif self.infer_mode == 'hard-EM':
                belief = self.graph.infer.inference( evidence = evid, 
                                            mode = 'max-product')
            else:
                raise Exception( 'Select the correct infer mode')
            
            # generate a complete data set with weight
            infer_datum = [[]]
            weights = [[1.]]
            for idx, state in enumerate(datum):
                # identify the node 
                node_name = self.graph.name_nodes[ idx]
                # if not in evidence 
                if node_name in belief.keys():
                    i_node = self.graph.get_node( node_name)
                    if self.infer_mode == 'EM':
                        new_datum = []
                        new_weights = []
                        # add all possible value of the node to the existed data and their weights
                        # e.g. [ a ] --> [ [a, b1], [a, b2], [a, b3]],
                        #      [ 1 ] --> [ [.4], [.4], [.2]]
                        for i_sample, w_sample in zip(infer_datum, weights):
                            for i_state, w in zip( i_node.states, belief[ node_name]):
                                new_datum.append( i_sample + [i_state])
                                new_weights.append( [ w * w_sample[0]])
                        infer_datum = new_datum
                        weights = new_weights
                    elif self.infer_mode == 'hard-EM':
                        new_datum = []
                        state = i_node.states[ belief[ node_name]]
                        for i_sample in infer_datum:
                            new_datum.append( i_sample + [state])
                        infer_datum = new_datum
                    else:
                        raise Exception( 'Select the correct infer mode')  
                # if is evidence: append the value to all exited data                  
                else:
                    new_datum = []
                    for i_sample in infer_datum:
                        new_datum.append( i_sample + [str(state)])
                    infer_datum  = new_datum

            # assign the infer data into a big list
            infer_data += infer_datum
            infer_weights += weights

        return np.array(infer_data), np.reshape( np.array(infer_weights), [-1])


'''
PART4: Bayes Net (beta version)

The "bayesNet" is the stable class I have developed in the previous class. 
However, in this project, we need some advanced functions. So I see the old
BayesNet function as parent class. I will test new code in this new class. 
Maybe in the next project, I will intergrate these new functions into my basic
file. 

The new functions allow the bayes net to:

* learn cpt 
* learn modify and learn the structure 
'''
class bayesNet_beta( bayesNet):

    def __init__( self):
        super( bayesNet_beta , self).__init__()

    def best_params( self, data, weights):
        return self.learner.best_params( data, weights)

    def cal_bic( self, data, weights=[]):
        '''Calculate likelihood

        Computing likelihood based on the complete data. Some of the 
        complete data is infered data, so we need to * weight.
        Xm = {ym, zm}. 

        * LL( D; theta) = \sum_xm LL( xm; theta)
        * LL( D; theta) = \sum_{zm,ym} w(zm,ym) log p(z=zm, y=ym|theta), 
        here I use the function fac_take to get p(z=m,y=m|theta)
           
        Here we used theta for both q and p, due to the convergence
        of EM.
        '''
        sum_logLike = 0.
        if len(weights)==0:
            weights = np.ones( [data.shape[0],])
        prob = None
        for datum, weight in zip(data, weights): 
            # clear the sample stored in the graph
            self.clear_samples()
            # get p(x|theta)
            for n, i_node_name in enumerate(self.name_nodes):
                i_node = self.get_node( i_node_name)
                if n == 0:
                    prob = i_node.cpt
                else:
                    prob = fac_prod( prob, i_node.cpt)
            # get log p(x=xm|theta)
            for i_node_name in self.name_nodes:
                node_idx = self.name_nodes.index( i_node_name)
                state_idx = self.get_node(i_node_name)._state_to_idx( datum[node_idx])
                prob = fac_take( prob, i_node_name, state_idx)
            log_prob = prob.log().get_distribution()
            # log_like = sum_m log (xm|theta)
            sum_logLike += weight * log_prob

        d = self.all_params().shape[0]
        M = 500
        bic = sum_logLike - d / 2 * np.log( M)
        return bic, sum_logLike
        
    def assign_algs( self, infer='belif_prop', 
                           learner = 'counter', 
                           struct_learner= 'hill_climb'):
        if infer == 'belif_prop':
            self.infer = inference.belief_prop( self)
        if learner == 'counter':
            self.learner = counter(self)
        if struct_learner == 'hill_climb':
            self.struct_learner = hill_climbing(self)

    def isloopy( self):
        '''Check if the structure has loop

        Cannot understand the Notear's DAG. So
        I create a by myself. The idea is if we follows the 
        children, and the children of the children recursivly
        and visit the same nodes, this means the bayes net has 
        loop. This is like traverse over a tree structure. 

        Get the idea from leetcode question 94.

        Ouput: 
            True or False
        '''
        # A recurisve function
        def visit_yet( i_node, visited_lst):
            # if the "new node" has been visted 
            if i_node in visited_lst:
                return True
            # record the visited node
            visited_lst.append( i_node)
            # follow the children
            if len(i_node.children):
                for child_node in i_node.children:
                    if visit_yet(child_node, list(visited_lst)):
                        return True 
            return False
        # empty list before visit
        for i_node_name in self.name_nodes:
            i_node = self.get_node( i_node_name)
            if visit_yet( i_node, list()):
                return True
        return False


if __name__ == "__main__":

    ############################################
    ####             LOAD DATA              ####
    ############################################

    dir = os.path.dirname(os.path.abspath(__file__))
    s = loadmat( dir + '/data/data_missing_500.mat')
    data = s['data_500'].astype(str) 
   
    ############################################
    ####       Build a empty binary BN      ####
    ############################################

    # init an empty BN 
    graph = bayesNet_beta()
    
    # add nodes to the graph, name, type, prior 
    graph.add_nodes(
        [ 'A', [ '1', '2']],
        [ 'B', [ '1', '2']],
        [ 'C', [ '1', '2']], 
        [ 'D', [ '1', '2']],
        [ 'E', [ '1', '2']])

    # init cpt as uniform 
    graph.init_params()
    graph.assign_algs()
    
    ############################################
    ####             Experiment             ####
    ############################################

    ## Exp1: Visualize the learned CPT table]
    SEM = struct_EM( graph)
    bics, logLikes = SEM.learn_structure_and_params( data)
    print( bics, logLikes)
    
    # plot the bics along with the SEM iteration
    plt.style.use( 'ggplot')
    plt.figure( figsize= [7, 6])
    x = range( 1, len(bics)+1)
    plt.plot( x, bics, linewidth = 2)
    plt.plot( x, logLikes, linewidth = 2)
    plt.legend( ['bic', 'logLike'])
    plt.xlabel( 'iterations')    
    plt.ylabel( 'score')
    plt.title( 'BIC and Loglike score over iterations')
    plt.savefig( dir + '/bic_SEM.png')

    # visualize CPT
    graph.vis_params()
    

