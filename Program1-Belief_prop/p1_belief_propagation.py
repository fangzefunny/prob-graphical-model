''' 
Project 1 Belif propagation 
Zeming Fang

IMPORTANT NOTE: dim0 means F, dim1 means T. 
This definition is to the opposite to the homework instruction
'''

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from collections import namedtuple
name_node = namedtuple( 'node', ('name', 'content'))

'''
# PART0: utility

In this part, we define the some utility functions. All functions falls
to one of the following two categories

* probability calculation: factor, fac_product, fac_sum, fac_max
    These functions in this category define how to operate the probability
    distribution. 

* transition between different data structure: prob_matrix, normalize

'''

class factor:
    '''
    all the probability distribution should coded as a factor,
    that store the information of variables and distribution. 
    '''
    def __init__(self, variables = None, distribution = None):
        if (distribution is None) and (variables is not None):
            self.__set_data(np.array(variables), None, None)
        elif (variables is None) or (len(variables) != len(distribution.shape)):
            raise Exception('Data is incorrect')
        else:
            self.__set_data( np.array(variables),
                             np.array(distribution),
                             np.array(distribution.shape))
    
    def __set_data(self, variables, distribution, shape):
        self.__variables    = variables
        self.__distribution = distribution
        self.__shape        = shape
        
    def is_none(self):
        return True if self.__distribution is None else False
        
    def get_variables(self):
        return self.__variables
    
    def get_distribution(self):
        return self.__distribution
    
    def get_shape(self):
        return self.__shape
    
def fac_prod(x, y):
    '''
    Implement of product between probability distribution.
    Can automatically do the variables matching.
    '''
    if x.is_none() or y.is_none():
        raise Exception('One of the factors is None')
    
    xy, xy_in_x_ind, xy_in_y_ind = np.intersect1d( x.get_variables(), y.get_variables(), return_indices=True)
    
    if xy.size == 0:
        raise Exception('Factors do not have common variables')
    
    if not np.all(x.get_shape()[xy_in_x_ind] == y.get_shape()[xy_in_y_ind]):
        raise Exception('Common variables have different order')
    
    x_not_in_y = np.setdiff1d(x.get_variables(), y.get_variables(), assume_unique=True)
    y_not_in_x = np.setdiff1d(y.get_variables(), x.get_variables(), assume_unique=True)
    
    x_mask = np.isin(x.get_variables(), xy, invert=True)
    y_mask = np.isin(y.get_variables(), xy, invert=True)
    
    x_ind = np.array([-1]*len(x.get_variables()), dtype=int)
    y_ind = np.array([-1]*len(y.get_variables()), dtype=int)
    
    x_ind[x_mask] = np.arange(np.sum(x_mask))
    y_ind[y_mask] = np.arange(np.sum(y_mask)) + np.sum(np.invert(y_mask))
    
    x_ind[xy_in_x_ind] = np.arange(len(xy)) + np.sum(x_mask)
    y_ind[xy_in_y_ind] = np.arange(len(xy))
    
    x_distribution = np.moveaxis(x.get_distribution(), range(len(x_ind)), x_ind)
    y_distribution = np.moveaxis(y.get_distribution(), range(len(y_ind)), y_ind)
                
    res_distribution =   x_distribution[tuple([slice(None)]*len(x.get_variables())+[None]*len(y_not_in_x))] \
                       * y_distribution[tuple([None]*len(x_not_in_y)+[slice(None)])]
    
    return factor(list(x_not_in_y)+list(xy)+list(y_not_in_x), res_distribution)

def fac_sum(x, variables):
    '''
    Implement of mariginal over probability distribution.
    '''

    variables = np.array( variables)
    
    if not np.all(np.in1d(variables, x.get_variables())):
        raise Exception('Factor do not contain given variables')
    
    res_variables    = np.setdiff1d(x.get_variables(), variables, assume_unique=True)
    res_distribution = np.sum(x.get_distribution(),
                              tuple(np.where(np.isin(x.get_variables(), variables))[0]))
    
    return factor(res_variables, res_distribution)

def fac_max(x, variables):
    '''
    Implement of maximization over probability distribution.
    '''
    variables = np.array( variables)
    
    if not np.all(np.in1d(variables, x.get_variables())):
        raise Exception('Factor do not contain given variables')

    res_variables    = np.setdiff1d(x.get_variables(), variables, assume_unique=True)
    res_distribution = np.max(x.get_distribution(),
                              tuple(np.where(np.isin(x.get_variables(), variables))[0]))

    return factor(res_variables, res_distribution)

def prob_matrix( cpt):
    '''
    Turn cpt into a matrix form
    '''
    lst = []
    for key in cpt.keys():
        lst += cpt[key]
    n_dim = int( np.log2( len(lst)))    
    return np.array(lst).reshape([2]*n_dim, order = 'F')

def normalize( dist):
    return dist/np.sum(dist)

'''
# PART1: Bayesian Network 

In this part, we define the class for building a BN.
Classes include:

* Node
* Graph
'''

class node:
    '''
    A class that defines nodes in a BN.
    This class store information includes:
    '''
    def __init__( self, name, prior = None):
        self.name           = name
        self.parents        = []
        self.children       = []
        self.cpt            = factor( [name], prior)
        self.prior_msg      = factor( [name], prior)
        self.likelihood_msg = factor( [name], None)
        self.is_evidence    = 0.
        
    def show_msg( self):
        print( 'For {}:\n pi_msg:     {},\n lambda_msg: {},\n is_evidence: {}'.format(
            self.name,
            np.round( self.prior_msg.get_distribution(), 4),
            np.round( self.likelihood_msg.get_distribution(), 4),
            self.is_evidence))
        
class BayesNet:
    '''
    A class that defines a BN graph. 
    In this class, you need to use .add_nodes, and .add_edges function
    to build the graph.
    '''
    
    def __init__( self):
        self.nodes = []
        self.edges = []
    
    def reset( self):
        for i_node in self.nodes:
            i_node.content.prior_msg = []
            i_node.content.likelihood_msg = []
            i_node.content.is_evid = []
        
    def name_to_node( self, name):
        return self.nodes[ self.name_nodes.index( name)].content
    
    def get_node( self, name):
        return self.name_to_node( name)

    def add_nodes( self, *args):
        for item in args:
            node_name, prior = item
            self.nodes += [ name_node( node_name, node( node_name, prior))] 
            
        # an axulliary list to help indexing
        self.name_nodes = list(name_node( *zip( *self.nodes)).name)
        
    def add_edges( self, *args):
        for item in args:
            from_, to_, cpt = item
            parents = [ self.get_node( name) for name in from_]
            child   = self.name_to_node( to_)
            for i_node in parents:
                i_node.children.append( child)
            child.parents += parents
            variables = [ child.name] + [ par_node.name for par_node in parents]
            child.cpt = factor( variables, prob_matrix( cpt))
        
    def sanity_check( self):
        '''
        Check if the code is bug-free 
        '''
        for node_name in self.name_nodes:
            i_node = self.get_node( node_name)
            print( 'For {}, parents: {}, children: {}'.format( 
                        i_node.name, 
                        str([par_node.name for par_node in i_node.parents]), 
                        str([child_node.name for child_node in i_node.children])))

'''
# PART3: Belief propagation

In this part, we define the algorithm of belief propagation
to do Inference
'''
            
class belief_prop:
    
    def __init__( self, graph, tol = 1e-3, max_epi = 10, verbose=False, show_plot=False):
        self.graph = graph
        self.sequence = None
        self.belief_table = {}
        self.tol = tol          # tolerance of convergence 
        self.max_epi = max_epi  # maximum epi before terminate the inference 
        self.mode = 'sum-product'
        self.infer = fac_sum
        self.verbose = verbose
        self.show_plot = show_plot
       
    def show_msg( self):
        '''
        show pi_msg and lambda_msg
        '''
        for node_name in self.graph.name_nodes:
            i_node = self.graph.get_node( node_name)
            i_node.show_msg()
            
    def init_belief_table( self):
        '''
        init belief table, prepare for the output 
        '''
        for i_node in self.sequence:
            self.belief_table[ i_node] = np.array([ 0, 0])    
    
    def fix_evidence( self, evid_str):
        '''
        input the evidence, and initiate the cached pi_msg and lambda_msg
        '''
        evid_str = np.array(list(evid_str))
        eq_idx = np.where( evid_str == '=')[0]
        evid_vars = evid_str[eq_idx-1]
        evid_vals = evid_str[eq_idx+1]
        for evid_name, evid_val in zip( evid_vars, evid_vals):
            evid_node = self.graph.get_node( evid_name)
            prior_msg, likelihood_msg = np.zeros([2,]), np.zeros([2,])
            prior_msg[ int( evid_val)] = 1.
            likelihood_msg[ int( evid_val)] = 1.
            evid_node.is_evidence = 1.
            evid_node.prior_msg = factor( [evid_name], prior_msg)
            evid_node.likelihood_msg = factor( [evid_name], likelihood_msg)

        node_list = self.graph.name_nodes.copy()
        for evid_name in evid_vars:
            node_list.pop( node_list.index(evid_name))
        
        if self.sequence == None:
            self.sequence = np.random.permutation( node_list)

        for node_name in node_list:
            i_node = self.graph.get_node( node_name)
            if len( i_node.parents) == 0:
                i_node.likelihood_msg = factor( [node_name], np.ones( [2,]))
            elif len( i_node.children) == 0:
                i_node.likelihood_msg = factor( [node_name], np.ones( [2,]))
                i_node.prior_msg = factor( [node_name], np.ones( [2,]))
            else:
                i_node.likelihood_msg = factor( [node_name], np.ones( [2,]))
                i_node.prior_msg = factor( [node_name], np.ones( [2,])) 
                
    def tot_prior_msg( self, query):
        '''
        Calculate: Pi(Xq)
        '''
        tot_prior_msg = query.cpt
        infer_over = []
        if len( query.parents):
            for parent in query.parents:
                infer_over += [parent.name]
                tot_prior_msg = fac_prod(tot_prior_msg, self.prior_msg_from_parent( parent, query))
        return self.infer( tot_prior_msg, infer_over)
    
    def prior_msg_from_parent( self, parent, target):
        '''
        Calculate: Pi_par(Target)
        '''
        parent_prior_msg = parent.prior_msg
        children_no_target = parent.children.copy()
        children_no_target.pop( children_no_target.index( target))
        tot_likelihood_msg = factor( [ parent.name], np.array([ 1., 1.]))
        if len( children_no_target):
            for child in children_no_target:
                tot_likelihood_msg = fac_prod( tot_likelihood_msg, 
                                              self.likelihood_msg_from_child( child, parent))
        return fac_prod( parent_prior_msg, tot_likelihood_msg)
    
    def tot_likelihood_msg( self, query):
        '''
        Calculate: Lambda(Xq)
        '''
        tot_likelihood_msg = factor( [ query.name], np.array([ 1., 1.]))
        if len( query.children):
            for child in query.children:
                tot_likelihood_msg = fac_prod( tot_likelihood_msg, 
                                               self.likelihood_msg_from_child( child, query))
        return tot_likelihood_msg
    
    
    def likelihood_msg_from_child( self, child, target):
        '''
        Calculate: Lambda_child(Target)
        '''
        child_likelihood_msg = child.likelihood_msg
        parents_no_target = child.parents.copy()
        parents_no_target.pop( parents_no_target.index(target))
        tot_prior_msg = child.cpt
        infer_over = []
        if len( parents_no_target):
            for parent in parents_no_target:
                infer_over += [parent.name]
                tot_prior_msg = fac_prod( tot_prior_msg, 
                                         self.prior_msg_from_parent( parent, child))
        return self.infer( fac_prod( child_likelihood_msg, self.infer( tot_prior_msg, infer_over)), child.name)

    def step( self, query_name):
        '''
        Visit a node, compute the belief and update cached pi_msg and lambda_msg
        '''
        cache_belief = self.belief_table[ query_name]
        query = self.graph.get_node( query_name)
        query.prior_msg = self.tot_prior_msg( query)
        query.likelihood_msg = self.tot_likelihood_msg( query)
        new_belief = normalize(fac_prod( query.prior_msg, query.likelihood_msg).get_distribution())
        abs_delta = abs( cache_belief - new_belief).sum()  
        if self.verbose:  
            query.show_msg()
            print( ' belief:     {}'.format(str(np.round(new_belief,4))))
        self.belief_table[ query_name] = new_belief
        return abs_delta 

    def show_convergence( self, deltas):
        plt.style.use('ggplot')
        plt.figure( figsize = [8, 6] )
        x = range( 1, len(deltas)+1)
        plt.plot( x, deltas, linewidth=2.)
        plt.xlabel( 'iterations')
        plt.ylabel( 'abs change of believies')
        plt.title( 'convergence conditions of max-product')
        plt.savefig( 'max-product.png' )
        
            
    def inference( self, evidence, mode = 'sum-product'):
        '''
        A unified function that summary the inference process,
        need to input evidence, and choose that kinds of inference
        you want to use
        '''
        if mode == 'sum-product':
            self.infer = fac_sum
        elif mode == 'max-product':
            self.infer = fac_max
        else:
            raise Exception( 'Please choose the correct inference method')
        
        self.fix_evidence( evidence )
        self.init_belief_table()

        if self.verbose:
            self.show_msg()

        if (self.sequence == None) and (len(self.sequence)==0):
            raise Exception('No unobservable nodes')
        
        # begin iteration
        done = 0. 
        epi = 0
        wait = 0.
        deltas = []
        while not done: 
            delta = 0.
            for i_node in self.sequence:
                delta += self.step( i_node)
            epi += 1
            deltas.append( delta)
            if (delta < self.tol):
                wait += 1  
            if (wait >= 3) or (epi >= self.max_epi):
                done = 1.
        if self.show_plot:
          self.show_convergence( deltas)

        # output 
        if mode == 'sum-product':
            return self.belief_table
        if mode == 'max-product':
            MPE = {}
            for key in self.belief_table:
                MPE[key] = np.argmax( self.belief_table[key])
            return MPE


if __name__ == "__main__":
    
    # p(S)
    prior_S = np.array( [ .2, .8])
    # p(A)
    prior_A = np.array( [ .4, .6])
    # p(L)
    prior_L = np.array( [ .2, .8])

    # hand code the conditinal probability table
    # p(B|S)
    cpd_BS = { 'S=0': [ .7, .3],
               'S=1': [ .3, .7]} 
    # p(T|A)
    cpd_TA = { 'A=0': [ .7, .3],
               'A=1': [ .2, .8]} 
    # p(E|L,T)
    cpd_ELT = { 'L=0,T=0': [ .9, .1],
                'L=1,T=0': [ .4, .6],
                'L=0,T=1': [ .3, .7],
                'L=1,T=1': [ .2, .8]}
    # p(F|B,E)
    cpd_FBE = { 'B=0,E=0': [ .8, .2],
                'B=1,E=0': [ .3, .7],
                'B=0,E=1': [ .2, .8],
                'B=1,E=1': [ .1, .9]}
    # p(X|E)
    cpd_XE = { 'E=0': [ .9, .1],
               'E=1': [ .2, .8]} 

    # init an empty BN 
    graph = BayesNet()

    # add nodes to the graph, name, type, prior 
    graph.add_nodes(
        [ 'S', prior_S],
        [ 'A', prior_A],
        [ 'B', None], 
        [ 'L', prior_L],
        [ 'T', None],
        [ 'E', None],
        [ 'F', None],
        [ 'X', None],
    )

    # add edges to the graph
    graph.add_edges( 
        [ [ 'S'], 'B', cpd_BS],
        [ [ 'A'], 'T', cpd_TA],
        [ [ 'L', 'T'], 'E', cpd_ELT],
        [ [ 'B', 'E'], 'F', cpd_FBE],
        [ [ 'E'], 'X', cpd_XE])
    # graph.sanity_check()
    
    # assign a bp algorithm 
    alg = belief_prop( graph, max_epi = 10)
    alg.sequence = [ 'S','A','B','L','T','E']
    ans = alg.inference( evidence = 'F=1,X=0', mode = 'sum-product')
    print( 'results of sum-product: \n', pd.DataFrame(ans))
    #alg.show_msg()

    # do max-product inference 
    graph.reset()
    alg = belief_prop( graph, max_epi = 10)
    alg.sequence = [ 'S','A','B','L','T','E']
    ans = alg.inference( evidence = 'F=1,X=0', mode = 'max-product')
    print( 'results of max-product: \n', pd.DataFrame(ans, index=['MPE']))
    #alg.show_msg() 

    #############################  Notes For grader #####################################
    #  
    # 1. Run the script and the results will be shown in the terminal window
    # 2. This definition of dimension is to the opposite to the homework instruction.
    #    In my code: dim0 means F, dim1 means T. 
    # 3. If you want to see the pi and lambda message after convergence,
    #    Uncomment the alg.show_msg() at line 489 and line 497 and rerun the script
    # 4. If you want to try new evidence, make sure do not include space in the string
    # 

    
