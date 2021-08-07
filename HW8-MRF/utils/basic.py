import numpy as np
from collections import namedtuple
from utils.prob_tool import *
from utils import inference 

name_node = namedtuple( 'node', ('name', 'content'))

class node:
    '''
    A class that defines nodes in a BN.
    This class store information includes:
    '''
    def __init__( self, name, states):
        # basic info 
        self.name           = name
        self.cpt            = None  
        self.states         = states 
        self.nstates        = len( states)
        # sample info
        self.curr_state     = None 
        self.curr_dist      = np.ones([self.nstates,])
        self.curr_idx       = None
        # relationships with other nodes 
        self.parents        = []
        self.children       = []
        self.markov_blanket = []
        # parent_configuration
        self.par_configs    = []
        # for evidence
        self.is_evid        = 0.

    def _init_cpt( self, params_init = 'uniform'):
        ncol = self.nstates
        nrow = 1. 
        vars_lst = [self.name] 
        dim_lst = [self.nstates]
        if len(self.parents):
            for par_node in self.parents:
                vars_lst += [par_node.name]
                dim_lst += [par_node.nstates]
                nrow *= par_node.nstates
        key = 0
        dist = dict()
        for _ in range(int(nrow)):
            if params_init == 'uniform':
                dist[ str(key)] = list(np.ones( [ ncol,]) / ncol)
            elif params_init == 'random':
                std = 2 / (1 + ncol)
                dist[ str(key)] = list( normalize( sigmoid(std * np.random.randn(ncol,))))
            else:
                raise Exception( 'Choose the correct initializer')
            key += 1
        self.cpt = factor( vars_lst, prob_matrix(dist, dim_lst))

    def _idx_to_state( self, idx):
        return self.states[ idx]

    def _state_to_idx( self, state):
        return self.states.index( state)
    
    def _idx_to_dist( self, idx):
        onehot = np.zeros( [self.nstates])
        onehot[ idx] = 1.
        return factor( [self.name], onehot)

    def _params_to_cpt( self, params):
        vars = [ self.name] + [ par_node.name for par_node in self.parents]
        nstates = [ self.nstates] + [ par_node.nstates for par_node in self.parents]
        self.cpt = factor( vars, prob_matrix( params, nstates))

    def _cpt_to_params( self):
        params = dict()
        i_params = np.reshape( self.cpt.get_distribution(), 
                               [-1, self.nstates], order='F')
        for row, par_config in enumerate(self.par_configs):
            params[ par_config] = list(i_params[ row, :])
        return params 

    def register_state( self, idx):
        self.curr_idx = idx
        self.curr_state = self._idx_to_state( idx)
        self.curr_dist = self._idx_to_dist( idx)
        
    def register_evid( self, state):
        self.is_eivd = 1.
        idx = self._state_to_idx( state)
        self.register_state( idx)
        
    def clear_sample( self):
        self.curr_state     = None 
        self.curr_dist      = np.ones([self.nstates,])
        self.curr_idx       = None

    def show_cpt( self):
        return self.cpt.show_dist()
        
class bayesNet:
    '''
    A class that defines a BN graph. 
    In this class, you need to use .add_nodes, and .add_edges function
    to build the graph.
    '''
    
    def __init__( self):
        self.nodes = []
        self.edges = []
        self.params = []

    def reset( self):
        for i_node in self.nodes:
            i_node.content.prior_msg      = []
            i_node.content.likelihood_msg = []
            i_node.content.is_evid        = []
            i_node.content.curr_state     = None
            i_node.content.curr_dist      = None
            i_node.content.curr_idx       = None
            #i_node.content._init_cpt()
            
    def clear_samples( self):
        for i_node in self.nodes:
            i_node.content.curr_state     = None
            i_node.content.curr_dist      = None
            i_node.content.curr_idx       = None

    def clear_struct( self):
        for i_node in self.nodes:
            i_node.content.parents        = []
            i_node.content.children       = []

    def name_to_node( self, name):
        return self.nodes[ self.name_nodes.index( name)].content



    def get_par_config( self):
        '''For 2d visualization of cpt

        In the 2d visualization of cpt, 
        * cols represent the state of the node;
        * rows represent the configuration of the parents

        This function is to create rows to enumerate all possible
        parent configurations  
        '''
        # a dictionary to store the parent configurate
        conditions = dict()
        for node_name in self.name_nodes:
            # get the target node 
            i_node = self.get_node( node_name)
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
                # if a node has no parents, we call it prior 
                par_config = 'prior'
                # if a node has parents, 
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
            i_node.par_configs = par_configs
    
    def get_node( self, name):
        return self.name_to_node( name)

    def add_nodes( self, *args):
        for item in args:
            node_name, value_idx = item
            self.nodes += [ name_node( node_name, node( node_name, value_idx))] 
            
        # an axulliary list to help indexing
        self.name_nodes = list(name_node( *zip( *self.nodes)).name)
        
    def add_edges( self, *args):
        for item in args:
            from_, to_ = item
            parent = self.get_node( from_)
            child  = self.get_node( to_) 
            parent.children.append( child)
            child.parents.append( parent)

    def remove_edges( self, *args):
        for item in args: 
            from_, to_ = item 
            parent = self.get_node( from_)
            child  = self.get_node( to_)
            parent.children.pop( parent.children.index(child))
            child.parents.pop( child.parents.index(parent))
    
    def reverse_edges( self, *args):
        for item in args:
            from_, to_ = item
            self.remove_edges( [ from_, to_])
            self.add_edges( [ to_, from_])

    def init_params( self, params_init = 'uniform'):
        for i_node in self.nodes:
            i_node.content._init_cpt( params_init)
        self.get_par_config()
    
    ## Load structure

    def load_struct( self, struct_lst):
        self.clear_struct()
        for struct in struct_lst:
            from_, to_ = struct[0], struct[-1]
            self.add_edges( [from_, to_])
        self.init_params()

    def load_params( self, params):
        '''From 2D params to factor cpt

        Input:
            params: a dictionary contains, a 2D params dict
            e.g for a 2D param dist
            p_AB = { 'B=0': [ .5, .5],
                     'B=1': [ .4, .6]} 

        '''
        for i_node_name in self.name_nodes:
            i_node = self.get_node( i_node_name)
            i_params = params[ i_node_name]
            i_node._params_to_cpt( i_params)

    ## Visualization  of the parameters and the structure 
    
    def print_params( self):
        params = dict()
        for i_node_name in self.name_nodes:
            i_node = self.get_node( i_node_name)
            i_params = i_node._cpt_to_params()
            params[ i_node_name] = i_params
        return params 

    def print_struct( self):
        '''Print the structure

        NEED debug 
        '''
        struct_lst = []
        for i_node_name in self.name_nodes:
            i_node = self.get_node( i_node_name)
            if len(i_node.parents):
                for par_node in i_node.parents:
                    relation_str = '{}-->{}'.format( 
                                   par_node.name, i_node_name)
                    if relation_str not in struct_lst:
                        struct_lst.append( relation_str)
            if len(i_node.children):
                for child_node in i_node.children:
                    relation_str = '{}-->{}'.format( 
                                  i_node_name, child_node.name)
                    if relation_str not in struct_lst:
                        struct_lst.append( relation_str)
        return struct_lst

    def vis_params( self):
        '''
        Check if the code is bug-free 
        '''
        for node_name in self.name_nodes:
            i_node = self.get_node( node_name)
            print( 'For {}, parents: {}, children: {} \n CPT:'.format( 
                        i_node.name, 
                        str([par_node.name for par_node in i_node.parents]), 
                        str([child_node.name for child_node in i_node.children])))
            i_node.cpt.show_dist()

    def all_params( self):
        params = []
        for i_node in self.nodes:
            flat_dist = np.reshape( i_node.content.cpt.get_distribution(),
                                    [-1], order = 'F')
            params += list(flat_dist)
        return np.array(params)


    

    
