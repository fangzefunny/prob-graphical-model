import os 
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import net, inference, normalize


class BayesNet( net):

    def __init__( self):
        super( BayesNet , self).__init__()

    def learn_cpt( self, data, mode ='mle', 
                               infer_mode = 'EM', 
                               params_init = 'uniform', 
                               verbose = False):
        if mode == 'mle':
            deltas = self.learner.mle( data, infer_mode = infer_mode, 
                                    params_init = params_init, 
                                    verbose = verbose)
            return deltas
        
    def fix_cpt( self, params):
        for node_name in self.name_nodes:
            i_node = self.get_node( node_name)
            i_node._params_to_cpt( params[ node_name])

    def assign_algs( self, infer='belif_prop', learn='counter' ):
        if infer == 'belif_prop':
            self.infer = inference.belief_prop( self)
        if learn == 'counter':
            self.learner = counter( self)
    
    def likelihood( self, data):
        self.learner.mle()



class counter:

    def __init__( self, graph, tol =1e-3, max_epi =20, verbose=False, infer_mode='EM'):
        self.graph = graph
        self.tol = tol
        self.max_epi = max_epi
        self.verbose = verbose
        self.infer_mode = infer_mode
        self.get_par_config()

    def _empty_counts( self):
        self.par_config = []

    def _split_data( self, data ):
        comp_data = []
        incomp_data = []
        for datum in data:    
            if np.sum( datum == '-999'):    
                incomp_data.append( datum)
            else:
                comp_data.append( datum)
        return np.array(comp_data), np.array(incomp_data)
    
    def get_par_config( self):
        conditions = dict()
        for node_name in self.graph.name_nodes:
            # get the target node 
            i_node = self.graph.get_node( node_name)
            cpt_dims = i_node.cpt.get_shape()
            par_lst = i_node.parents
            rows = 1.
            if len(par_lst):
                for dim in cpt_dims[1:]:
                    rows *= dim
            rows = int(rows)
            # create parent config name 
            par_configs = []
            for row in range(int(rows)):
                res = row 
                par_config = 'prior'
                if len(par_lst):
                    par_config = ''
                    for par_node in par_lst:
                        par_dim = par_node.nstates
                        par_name = par_node.name
                        par_state_idx = res % par_dim
                        res = res // par_dim
                        par_state = par_node._idx_to_state( par_state_idx)
                        par_config += '{}={},'.format( par_name, par_state)
                par_configs.append( par_config)
            
            conditions[ node_name] = par_configs
        self.par_configs = conditions

    def count_data( self, data, weights=[]):
        counts = {}
        for node_name in self.graph.name_nodes:
            node_idx = self.graph.name_nodes.index( node_name)
            i_node = self.graph.get_node( node_name)
            par_config = self.par_configs[ node_name]
            count_mat = np.zeros( [ len(par_config), i_node.nstates])
            # start filtering data
            for row, config_j in enumerate(par_config):
                config_str = np.array(list(config_j))
                eq_idx = np.where( config_str == '=')[0]
                par_names = config_str[eq_idx-1]
                par_states = config_str[eq_idx+1]
                data_filter = 1.
                if len(par_names):
                    for par_name, par_state in zip( par_names, par_states):
                        par_idx = self.graph.name_nodes.index( par_name)
                        data_filter *= ( data[ :, par_idx] == par_state)
                        for col, state in enumerate(i_node.states):
                            meet_requirements = data_filter * ( data[ :, node_idx] == state) 
                            if len(weights):
                                count_mat[ row, col] = np.sum( meet_requirements * weights)
                            else:
                                count_mat[ row, col] = np.sum( meet_requirements)
                else:
                    for col, state in enumerate(i_node.states):
                        meet_requirements = data_filter * ( data[ :, node_idx] == state) 
                        if len(weights):
                            count_mat[ row, col] = np.sum( meet_requirements * weights)
                        else:
                            count_mat[ row, col] = np.sum( meet_requirements)
            counts[ node_name] = count_mat
        return counts

    def count_to_params( self, counts):
        
        params = dict()
        for node_name in self.graph.name_nodes:
            par_config = self.par_configs[ node_name]
            cpt_params = dict()
            for row, config_j in enumerate( par_config):
                cpt_params[ config_j] = list(normalize( counts[node_name][ row, :]))
            params[ node_name] = cpt_params
        return params

    def mle( self, data, infer_mode = 'EM', params_init = 'uniform', verbose = False):
        self.infer_mode = infer_mode
        self.verbose = verbose
        self.params_init = params_init
        comp_data, incomp_data = self._split_data( data)
        comp_counts = self.count_data( comp_data)
        if incomp_data.shape[0] == 0:
            params = self.count_to_params( comp_counts)
            self.graph.fix_cpt( params)
        else:
            deltas = self.learn_incomp_data( incomp_data, comp_counts)
            return deltas


    def learn_incomp_data( self, incomp_data, comp_counts):
        '''
        Here we used EM algorithm
        '''
        done = False
        if self.params_init == 'comp_data':
            params = self.count_to_params( comp_counts)
            self.graph.fix_cpt( params)
        elif self.params_init == 'unifrom':
            self.graph.init_params( self.params_init)
        elif self.params_init == 'random':
            self.graph.init_params( self.params_init)
        elif self.params_init == 'local_search':
            self.graph.init_params( self.params_init)

        curr_params = self.graph.print_params()
        epi = -1
        deltas = []
        while not done:
            epi += 1
            # register the old params 
            old_params = curr_params
            # E-step
            infer_data, weights = self._infer_miss_data( incomp_data)
            infer_counts = self.count_data( infer_data, weights)
            # M-step
            counts = dict()
            for key in comp_counts:
                counts[key] = infer_counts[key] + comp_counts[key]
            params = self.count_to_params( counts)
            self.graph.fix_cpt( params)
            # check convergence
            curr_params = self.graph.print_params()
            delta = np.sum(abs(curr_params - old_params))
            deltas.append( delta)
            if delta < self.tol:
                done = True
            if self.verbose:
                print( 'At epi {}, the detla {:.4f}'.format( epi, delta))
        return deltas

    def _infer_miss_data( self, incomp_data):
        infer_data = []
        infer_weights = []
        for t in range(incomp_data.shape[0]):
            datum = incomp_data[t, :]
            # form evidence
            evid = ''
            for idx, state in enumerate(datum):
                if state == '-999':
                    pass
                else:
                    node_name = self.graph.name_nodes[ idx]
                    evid += '{}={},'.format( node_name, state)
            
            self.graph.infer.reset()
            if self.infer_mode == 'EM':
                belief = self.graph.infer.inference( evidence = evid, 
                                            mode = 'sum-product')
            elif self.infer_mode == 'hard-EM':
                belief = self.graph.infer.inference( evidence = evid, 
                                            mode = 'max-product')
            else:
                raise Exception( 'Select the correct infer mode')
            
            # fill data with weight
            infer_datum = [[]]
            weights = [[1.]]
            for idx, state in enumerate(datum):
                node_name = self.graph.name_nodes[ idx]
                if node_name in belief.keys():
                    i_node = self.graph.get_node( node_name)
                    if self.infer_mode == 'EM':
                        new_datum = []
                        new_weights = []
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
                else:
                    new_datum = []
                    for i_sample in infer_datum:
                        new_datum.append( i_sample + [str(state)])
                    infer_datum  = new_datum
            # assign the infer data into a big list
            infer_data += infer_datum
            infer_weights += weights

        return np.array(infer_data), np.reshape( np.array(infer_weights), [-1])
          
if __name__ == "__main__":

    ############################################
    ####             LOAD DATA              ####
    ############################################

    dir = os.path.dirname(os.path.abspath(__file__))
    s = loadmat( dir + '/data/discrete_missing_500.mat')
    data = s['discrete_missing_500'].astype(str)   
   
    ############################################
    ####   Build a BN with no paramter      ####
    ############################################

    # init an empty BN 
    graph = BayesNet()
    # add nodes to the graph, name, type, prior 
    graph.add_nodes(
        [ 'A', [ '1', '2']],
        [ 'B', [ '1', '2', '3']],
        [ 'C', [ '1', '2', '3']], 
        [ 'D', [ '1', '2']],
        [ 'E', [ '1', '2']],
        [ 'F', [ '1', '2']])
    # add edges to the graph
    graph.add_edges( 
        [ [ ], 'A', []],
        [ [ ], 'B', []],
        [ [ 'A', 'B'], 'D', []],
        [ [ 'A'], 'C', []],
        [ [ 'C', 'F'], 'E', []],
        [ [ 'A', 'D'], 'F', []])
        
    # fix inference and learn algorithm
    graph.assign_algs()

    ############################################
    ####             Experiment             ####
    ############################################

    ## Exp1: Visualize the learned CPT table

    # show CPT of EM
    _ = graph.learn_cpt( data,
                        infer_mode='EM', 
                        params_init='uniform', 
                        verbose=True)
    graph.vis_params()

    # show CPT of hard-EM
    graph.init_params()
    _ = graph.learn_cpt( data,
                        infer_mode='hard-EM', 
                        params_init='uniform', 
                        verbose=True)
    graph.vis_params()


    ## Exp 2: compare EM and hard EM with uniform init 

    # check EM with uniform init
    fill_methods = [ 'EM', 'hard-EM']
    legend_lst = fill_methods
    color_lst = [ 'r', 'b', 'g', 'orange', 'k']
    record = np.zeros( [ len(fill_methods), 10])
    for row, fill_method in enumerate(fill_methods):
        graph.init_params()
        deltas = graph.learn_cpt( data,
                                 infer_mode=fill_method, 
                                 params_init='uniform', 
                                 verbose=False)
        record[ row, :len(deltas)] = deltas
        #print(deltas )

    plt.style.use( 'ggplot')
    plt.figure( figsize = [ 7, 5])
    for n in range( len(fill_methods)):
        plt.plot( np.arange(0,10), record[ n, : ],
                  marker = 'o', linewidth=2., color = color_lst[n] )
    plt.xlabel( 'iteration')
    plt.ylabel( 'abs change of parameters')
    plt.title( 'EM vs Hard-EM')
    plt.legend( legend_lst)
    #plt.ylim( [0, .25] )
    plt.savefig( dir + '/EM_vs_hard-EM_uniform' )
    
    ## Exp 3: EM and hard EM with different init method 

    # check EM with uniform init
    fill_methods = [ 'EM', 'hard-EM']
    init_methods = [ 'uniform', 'random', 'random', 'comp_data']
    legend_lst = [ 'uniform', 'random1', 'random2', 'comp_data']
    color_lst = [ 'r', 'b', 'g', 'orange', 'k']
    record = np.zeros( [ len(init_methods), 10])
    for row, init_method in enumerate(init_methods):
        graph.init_params()
        deltas = graph.learn_cpt( data,
                                 infer_mode='EM', 
                                 params_init=init_method, 
                                 verbose=False)
        record[ row, :len(deltas)] = deltas
        #print(deltas )

    plt.style.use( 'ggplot')
    plt.figure( figsize = [ 7, 5])
    for n in range( len(init_methods)):
        plt.plot( np.arange(0,10), record[ n, : ],
                  marker = 'o', linewidth=2., color = color_lst[n] )
    plt.xlabel( 'iteration')
    plt.ylabel( 'abs change of parameters')
    plt.title( 'Different init method-EM')
    plt.legend( legend_lst)
    #plt.ylim( [0, .25] )
    plt.savefig( dir + '/EM_with_different_inits' )
    


    
    
    
    
    
    
    
    
    
    




    