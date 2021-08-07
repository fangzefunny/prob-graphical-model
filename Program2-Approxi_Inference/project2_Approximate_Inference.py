import time
import pickle 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
from collections import OrderedDict, namedtuple, Counter

'''
Project 2 Approximate Inference 

'''

###########################################
#                                         #
#           PART1: Utils                  #
#                                         #
###########################################

name_node = namedtuple( 'node', ('name', 'content'))

class factor:
    '''
    all the probability distribution should coded as a factor,
    that store the information of variables and distribution.

    example: 
    pA = factor( [ 'A'], np.array( [0.5, 0.5]) ) 
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
        self.get_variables 
        return self.__distribution

    def log( self):
        return factor( self.__variables, np.log( self.__distribution + 1e-13))
    
    def get_shape(self):
        return self.__shape
    
    def show_distribution(self):
        dist = self.get_distribution()
        var  = self.get_variables() 
        n_dim = len( var)  
        dist_lst = dist.reshape([-1], order='F')
        out_dict = { var[0]+'=0': [],var[0]+'=1': [] }
        indices = []
        start_idx = 0 
        value_space = [0, 1]
        while start_idx < len( dist_lst): 
            row_idx = ''
            for pos, i_var in enumerate(var[1:]):
                i_value = to_bin( int(start_idx/2), n_dim-1)[-(pos+1)]
                row_idx += i_var + '=' + i_value
            indices.append( row_idx)
            for t_value in value_space:
                out_dict[ var[0]+'='+ str(t_value)].append( dist_lst[ start_idx])
                start_idx += 1
        return pd.DataFrame(out_dict,index=indices)
        
def fac_prod(x, y):
    '''Factor product 
    Implement of product between probability distribution.
    Can automatically do the variables matching.

    Example: compute p(A)p(B)
    
    p_AB = fac_prod( pA, pB)
    '''
    if x.is_none() or y.is_none():
        raise Exception('One of the factors is None')
    
    xy, xy_in_x_ind, xy_in_y_ind = np.intersect1d( x.get_variables(), y.get_variables(), return_indices=True)
    
    if xy.size == 0:
        val, dim = y.get_variables()[0], y.get_shape()[0]
        dist = np.expand_dims( x.get_distribution(), axis = -1)
        new_dist = np.tile( dist, [1, dim])
        new_x = factor( list(x.get_variables())+[val], new_dist)
        return fac_prod( new_x , y)
        
    
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
    
def fac_div(x, y):
    '''Factor divide
    Implement of product between probability distribution.
    Can automatically do the variables matching.

    Example: compute p(A)/p(B)
    
    p_AB = fac_div( pA, pB)
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
                       / y_distribution[tuple([None]*len(x_not_in_y)+[slice(None)])]
    
    return factor(list(x_not_in_y)+list(xy)+list(y_not_in_x), res_distribution)

def fac_sum(x, variables=[]):
    '''Factor marginalization 
    Implement of mariginal over probability distribution.

    Example: compute sum_{A} p(A, B)
    
    p_B = fac_sum( p_AB, ['A'])
    '''

    variables = np.array( variables)
    
    if not np.all(np.in1d(variables, x.get_variables())):
        raise Exception('Factor do not contain given variables')
    
    res_variables    = np.setdiff1d(x.get_variables(), variables, assume_unique=True)
    res_distribution = np.sum(x.get_distribution(),
                              tuple(np.where(np.isin(x.get_variables(), variables))[0]))
    
    return factor(res_variables, res_distribution)

def fac_max(x, variables):
    '''Maximum explantation
    Implement of maximization over probability distribution.

    Example: compute max_{A} p(A, B)

    fac_max( p_AB, ['A'])
    '''
    variables = np.array( variables)
    
    if not np.all(np.in1d(variables, x.get_variables())):
        raise Exception('Factor do not contain given variables')

    res_variables    = np.setdiff1d(x.get_variables(), variables, assume_unique=True)
    res_distribution = np.max(x.get_distribution(),
                              tuple(np.where(np.isin(x.get_variables(), variables))[0]))

    return factor(res_variables, res_distribution)

def fac_take(x, variable, value):
    '''Choose a value 
    I do not use this function often. I prefer sum_B I(B=1) p(A,B) 
    
    Example, p(A, B=1):
    fac_take( p_AB, ['B'], 1)
    '''
    if x.is_none() or (variable is None) or (value is None):
        raise Exception('Input is None')
    
    if not np.any(variable == x.get_variables()):
        raise Exception('Factor do not contain given variable')
    
    if value >= x.get_shape()[np.where(variable==x.get_variables())[0]]:
        raise Exception('Incorrect value of given variable')
    
    res_variables    = np.setdiff1d(x.get_variables(), variable, assume_unique=True)
    res_distribution = np.take(x.get_distribution(),
                               value,
                               int(np.where(variable==x.get_variables())[0]))
    
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

def to_bin(x, ndim):
    a = bin(x)
    num0 = ndim + 2 - len(a)
    return a.replace('0b', '' + '0' * num0)

def _sample( prob):
    if abs(np.sum(prob) - 1) > 1e-6:
        raise Exception('The probability distribution should sum to 1.')
    u = np.random.rand()
    cat_idx = 0
    cdf = 0.
    for p in prob:
        cdf += p
        if u < cdf:
            return cat_idx
        cat_idx += 1 

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
    def __init__( self, name, states):
        self.name           = name
        self.cpt            = None  
        self.states         = states 
        self.nstates        = len( states)
        self.curr_state     = None  # sample state
        self.curr_dist      = None  # one hot distritbuion to show sample
        self.curr_idx       = None  # the index of the state
        # relationships with other nodes 
        self.parents        = []
        self.children       = []
        self.markov_blanket = []
        # for belief propagation
        self.prior_msg      = []
        self.likelihood_msg = []
        self.is_evid        = 0.

    # the map between state, index & one hot distribution
    def _idx_to_state( self, idx):
        return self.states[ idx] 

    def _state_to_idx( self, state):
        return self.states.index( state)
    
    def _idx_to_dist( self, idx):
        onehot = np.zeros( [self.nstates])
        onehot[ idx] = 1.
        return factor( [self.name], onehot)
    
    # assign a value to a certain node as evidence 
    def register_state( self, idx):
        self.curr_idx = idx
        self.curr_state = self._idx_to_state( idx)
        self.curr_dist = self._idx_to_dist( idx)
        
    def register_evid( self, state):
        self.is_eivd = 1.
        idx = self._state_to_idx( state)
        self.register_state( idx)
    
    # clear the cached value in the node 
    def clear_sample( self):
        self.curr_state     = None 
        self.curr_dist      = None
        self.curr_idx       = None

        
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
        '''Clear the cached value, the cached msg.

        This function is designed to init the belief propagation algorithm. 
        The name of this function need to "reconsidered" in the future.
        '''
        for i_node in self.nodes:
            i_node.content.prior_msg      = []
            i_node.content.likelihood_msg = []
            i_node.content.is_evid        = []
            i_node.content.curr_state     = None
            i_node.content.curr_dist      = None
            i_node.content.curr_idx       = None
            
    def clear_samples( self):
        '''Clear the cached value, the cached msg.

        This function is designed to init the sampler. 
        '''
        for i_node in self.nodes:
            i_node.content.curr_state     = None
            i_node.content.curr_dist      = None
            i_node.content.curr_idx       = None
        
    def name_to_node( self, name):
        return self.nodes[ self.name_nodes.index( name)].content
    
    def get_node( self, name):
        '''Map from name to the node
        '''
        return self.name_to_node( name)

    def add_nodes( self, *args):
        for item in args:
            node_name, value_idx = item
            self.nodes += [ name_node( node_name, node( node_name, value_idx))] 
            
        # an axulliary list to help indexing
        self.name_nodes = list(name_node( *zip( *self.nodes)).name)
        
    def add_edges( self, *args):
        '''Add links and meanwhile init cpt 
        '''
        for item in args:
            from_, to_, cpt = item
            parents = [ self.get_node( name) for name in from_]
            child   = self.name_to_node( to_) 
            for i_node in parents:
                i_node.children.append( child)
            child.parents += parents
            variables = [ child.name] + [ par_node.name for par_node in parents]
            nstates = [ child.nstates] + [ par_node.nstates for par_node in parents]
            child.cpt = factor( variables, cpt.reshape( nstates))
        
    def sanity_check( self):
        '''Check if the code is bug-free 
        '''
        for node_name in self.name_nodes:
            i_node = self.get_node( node_name)
            print( 'For {}, parents: {}, children: {}'.format( 
                        i_node.name, 
                        str([par_node.name for par_node in i_node.parents]), 
                        str([child_node.name for child_node in i_node.children])))


###########################################
#                                         #
#           PART2: Algorithms             #
#                                         #
###########################################

class base_sampler:
    '''High level knowledge about sampling
    '''
    def __init__( self, graph, seq=None):
        self.graph = graph
        self.seq   = seq
        self._setup_storages()
        self.clear_samples()

    def clear_samples(self):
        self.all_samples = []
          
    def _setup_storages( self):
        raise NotImplementedError 
        
    def sampling( self, n_samples, kwargs):
        raise NotImplementedError

    def show_samples( self,):
        '''See what is the value stored in each nodes
        '''
        for i_node in self.graph.nodes:
            print( i_node.name, 
                   i_node.content.curr_idx,
                   i_node.content.curr_state,
                   i_node.content.curr_dist.get_variables(),
                   i_node.content.curr_dist.get_distribution())

    def register_evid( self, conditions):
        '''Choose evidence E
        '''
        for node_name in conditions.keys():
            i_node = self.graph.get_node( node_name)
            i_node.register_evid( conditions[node_name])
        
    def posterior(self, query, all_samples):
        raise NotImplementedError
        
    def MAP( self, query, all_samples):
        i_node = self.graph.get_node( query)
        return i_node.states[ np.argmax( self.posterior( query, all_samples))]
    
    def infer( self, query, mode = 'posterior', n_samples = 1000, **kwargs):
        
        # collect samples
        if len(self.all_samples)==0:         
            self.all_samples = self.sampling( n_samples, kwargs)
        
        # calculate the normalized hitogram as infrence 
        if mode == 'posterior':
            return self.posterior( query, self.all_samples)
        elif mode == 'MAP':
            return self.MAP( query, self.all_samples)
        else:
            raise Exception( 'Choose the correct inference method')

'''
likelihood weighted sampling
'''

class likelihood_weighted_sampler(base_sampler):
    
    def __init__( self, graph, seq = None):
        super( likelihood_weighted_sampler, self).__init__( graph, seq)
        
    def _setup_storages( self):
        '''Apart from samplers, we also need weight
        '''
        self.Sample = namedtuple( 'Sample', self.seq + [ 'weight'])

    def prob_given_parents( self, i_node, target = None):
        '''Calculate p(Xi|pi(Xi))
        
        args:
            i_node: Xi a arbitrary node
            target (optional): some of the parent node might
                not have values. 

        output:
            A probability distribution: p(Xi|pi(Xi))
        '''
        prob   = i_node.cpt
        sum_over = []
        if len(i_node.parents):
            for par_node in i_node.parents:
                if par_node.name != target:
                    # sum over parent
                    sum_over += [par_node.name]
                    prob = fac_prod( prob, par_node.curr_dist)
        return fac_sum( prob, sum_over)
        
    def sampling( self, n_samples, conditions):
        '''Generate samples that meet the conditions

        args:
            n_samples: num of samples we need for this samplers
            conditions: evidence 

        output: 
            all_samples: a library of samples 
        '''
        # register evidence
        self.register_evid( conditions)
        self.graph.clear_samples()

        # start sampling
        all_samples = []
        for _ in range(n_samples):
            lst = []
            weight = 1.
            for node_name in self.seq:
                i_node = self.graph.get_node( node_name)
                prob = self.prob_given_parents( i_node
                            ).get_distribution()
                if i_node.is_evid:
                    a_sample = i_node.curr_idx
                    w = prob[ a_sample]
                    weight *= w
                else:
                    a_sample = _sample( prob)
                    i_node.register_state( a_sample)
                lst.append( a_sample)
            
            all_samples += [ self.Sample._make(lst + [weight])]

        return all_samples 
        
    def posterior(self, query, all_samples):
        '''Calculate p(query|evidence)
        '''
        summary = np.array( self.Sample( *zip( *all_samples))[ self.seq.index( query)]) 
        weight = np.array( self.Sample( *zip( *all_samples)).weight)
        dist = []
        for state in range(self.graph.get_node( query).nstates):
            dist.append( np.sum( weight * ( summary == state)))
        return normalize(dist)

'''
Gibbs sampling
'''

class gibbs_sampler( base_sampler):
    
    def __init__( self, graph, burnin = 5000, skip_step = 4):
        super( gibbs_sampler, self).__init__( graph, None)
        self.burnin = burnin
        self.skip_step = skip_step
        
    def _setup_storages( self):
        self.Sample = namedtuple( 'Sample', self.graph.name_nodes)
        
    def register_evid( self, conditions):
        '''Fix the evidence
        '''
        self.seq = self.graph.name_nodes.copy()
        for node_name in conditions.keys():
            i_node = self.graph.get_node( node_name)
            i_node.register_evid( conditions[node_name])
            self.seq.pop( self.seq.index( node_name))
            
    def init_chain( self):
        '''Init the markov chain

        Need to work on: different types of initiation 
        '''
        for node_name in self.graph.name_nodes:
            i_node = self.graph.get_node( node_name)
            if i_node.is_evid: 
                pass 
            else:
                uni = np.ones( [i_node.nstates,]) / i_node.nstates
                idx = _sample( uni)
                i_node.register_state(idx)
                
    def show_samples( self,):
        for i_node in self.graph.nodes:
            print( i_node.name, 
                   i_node.content.curr_idx,
                   i_node.content.curr_state,
                   i_node.content.curr_dist.get_variables(),
                   i_node.content.curr_dist.get_distribution())
    
    def prob_given_markov_blanket( self, i_node, target=None):
        '''Calculate p(Xi|pi(Xi)) \inverse_prod_{Yj \in child(X_i)} p(Yj|pi(Yj)
        
        args:
            i_node: Xi a arbitrary node
            target (optional): some of the parent node might
                not have values. 

        output:
            A probability distribution: p(Xi|MB(Xi))

        Warning & future work:

            Note1: 
            inverse_prod is very special, this is the basical the reason
            why I add a "target" input in the prob_given_parents function

            Note2: 
            p(X=k) can be implemented as "sum_x I(x=k) p(X)", 
            where I(x=k) is a one hot distribution: [ 0, 0, 0, 1, 0, 0]
        '''
        # p(Xi|pi(Xi))
        prob   = self.prob_given_parents( i_node)
        sum_over = []

        # p(Yj|pi(Yj)), Yj is the children of Xi
        if len( i_node.children):
            for child in i_node.children:
                sum_over += [child.name]
                prob = fac_prod( prob, child.curr_dist)
                prob = fac_prod( prob, 
                self.prob_given_parents( child, i_node.name))
        return fac_sum( prob, sum_over)
                
    def prob_given_parents( self, i_node, target = None):
        '''Calculate p(Xi|pi(Xi))
        
        args:
            i_node: Xi a arbitrary node
            target (optional): some of the parent node might
                not have values. 

        output:
            A probability distribution: p(Xi|pi(Xi))
        '''
        prob   = i_node.cpt
        sum_over = []
        if len(i_node.parents):
            for par_node in i_node.parents:
                if par_node.name != target:
                    sum_over += [par_node.name]
                    prob = fac_prod( prob, par_node.curr_dist)
        return fac_sum( prob, sum_over)
      
    def sampling( self, n_samples, conditions):
        '''Sampling main function

        This is the function I implement pseudo code

        '''

        # register the envidence nodes
        self.register_evid( conditions)
        self.init_chain()
        
        # uniform distribution for sampling the node
        uni = np.ones( [len( self.seq),]) / len( self.seq)
        
        # get the markov chain to burnin
        t = 0.
        burnin = True
        while burnin:
            
            # choose a random node xi ~ uniform
            node_name = self.seq[ _sample( uni)]
            i_node = self.graph.get_node( node_name)
            
            # calculat p(Xi|MB(Xi))
            prob_MB = normalize( self.prob_given_markov_blanket( i_node
                                ).get_distribution())

            # sample: Xi = k, k ~ p(Xi|MB(Xi))
            sample_idx = _sample( prob_MB)
            i_node.register_state( sample_idx)

            # add 1 step 
            t += 1 
            
            # check burin 
            if t > self.burnin:
                burnin = False
                
        # collect the first sample
        all_samples = []
        lst = [ i_node.content.curr_idx for i_node in 
              self.graph.nodes]
        all_samples += [ self.Sample._make(lst)]
        
        #start sampling
        for t in range(1, n_samples + 1):

            # choose a random node xi ~ uniform
            node_name = self.seq[ _sample( uni)]
            pos = self.graph.name_nodes.index( node_name) 
            i_node = self.graph.get_node( node_name)

            # calculat p(Xi|MB(Xi))
            prob_MB = normalize( self.prob_given_markov_blanket( i_node
                                ).get_distribution())

            # calculat p(Xi|MB(Xi))                    
            sample_idx = _sample( prob_MB)
            i_node.register_state( sample_idx)

            # a new sample !!I found the bug !!!!!! cant use list, 
            # turn it to np.array
            lst[ pos] = sample_idx 
    
            if t % self.skip_step == 0:
                all_samples += [ self.Sample._make(lst)]
                
        return all_samples
                
    def posterior(self, query, all_samples):
        idx = self.graph.name_nodes.index( query)
        summary = self.Sample( *zip( *all_samples))[idx]
        nstates = self.graph.get_node( query).nstates
        dist = np.zeros( [ nstates,])
        for x in summary:
            dist[x] += 1
        return normalize( dist)
            
'''
Variational Method
'''

class variation_method:
    
    def __init__( self, graph, tol = 1e-3, max_epi = 200, wait=3, 
                        verbose =False, show_plot =False):
        # init some hyperparamter
        self.graph = graph
        self.tol   = tol 
        self.max_epi = max_epi
        self.verbose = verbose
        self.show_plot = show_plot 
        self.wait   = wait
        # init a new mean field graph: what is we use 
        self._build_graph_mf()
        
    def reset( self):
        '''Init a approximator of the ground truth PGM

        Maybe this name is not appropriate. 
        A little overlap with _build_graph_mf function.
        '''
        self._build_graph_mf()
        
    def _build_graph_mf( self):
        '''Init a approximator of the ground truth PGM
        '''
        self.graph_mf = BayesNet()
        for node_name in self.graph.name_nodes:

            # build a node in the new graph that 
            node_true = self.graph.get_node( node_name)
            self.graph_mf.add_nodes([ node_name, node_true.states])
            node_mf = self.graph_mf.get_node( node_name)

            # init params as uniform 
            # (max entropy distribution in the discrete condition)
            init_parameters = np.ones( [ node_mf.nstates]) \
                                * 1 / node_mf.nstates 
            node_mf.cpt = factor( [ node_name], init_parameters)
            
    def register_evid( self, conditions):
        '''Fix the evidence

        Tell the variational approximator which of them are
        evidence. 
        '''
        for key in conditions.keys():
            node_mf = self.graph_mf.get_node( key)
            node_mf.is_evid = 1.
            # use indicator function as the cpt for the evidence node
            init_parameters = np.zeros( [ node_mf.nstates])
            state_idx = node_mf._state_to_idx( conditions[key])
            init_parameters[ state_idx] = 1.
            node_mf.cpt = factor( [key], init_parameters)
            
    def show_params( self):
        '''Show the params in the graph 

           Comment: in the wrong place,
                    consider to remove them
        '''
        for node in self.graph_mf.nodes:
            print( node.name, node.content.cpt.get_variables(),
                  node.content.cpt.get_distribution())

    def infer( self, query, mode = 'posterior', **kwargs):
        '''Do inference

        Where pseudo code is implemented

        args:
            query: the node you want to know
            mode: the type of inference you want to do
            kwargs: insert the evidence. 

        output: 
            inference result either:
            * p(query|evidence), or
            * max p(query|evidence)
        '''
        # tell the inference algorithm which of them are evidence. 
        self.register_evid( kwargs)
        
        #self.show_params()
        done = 0.
        epi  = 0.
        wait = 0.
        deltas = []
        while not done:
            delta = 0. 
            # for each node that is not evidence 
            for node_name in self.graph_mf.name_nodes:
                i_node = self.graph_mf.get_node( node_name)
                if i_node.is_evid:
                    pass
                else:
                    delta += self.update_params( node_name)
            epi += 1
            deltas.append( delta)
            if self.verbose:
                print( 'Iteration: {}, loss: {}'.format(int(epi), np.round(delta,4)))
            if (delta < self.tol):
                wait += 1 
            if ( wait >= self.wait) or (epi >= self.max_epi):
                done = 1.
                
        if self.show_plot:
            self.show_convergence(deltas, query)
        
        # do inference using the new mean-field graph 
        query_node = self.graph_mf.get_node( query)
        posterior = query_node.cpt.get_distribution()
        if mode == 'posterior':
            return posterior
        if mode == 'MAP':
            return query_node._idx_to_state( np.argmax( posterior))
                  
            
    def update_params( self, target_name):
        '''Update all params (theta_i) in a node Xi, 

        Theta i should be updated by iteratively 
        update the param theta_ik in each state. 

        args: 
            name of the node to index 

        output: 
            delta: absolute change of parameter in one node 
        '''
        
        t_node_mf = self.graph_mf.get_node( target_name)
        old_params = t_node_mf.cpt.get_distribution().copy()
        
        # parameter list (bad name, it should be logits)
        new_params = []
        
        for state_idx in range(len(t_node_mf.states)):
            # param_ik (bad name, it should be logits)
            param = self.step( t_node_mf, state_idx)
            new_params.append( param)
        
        # calculate the new paramters base
        new_params = normalize(np.exp(new_params))

        # replace the old param in the graph 
        t_node_mf.cpt = factor( [target_name], new_params)

        return np.sum( abs(new_params - old_params))
    
    def step( self, t_node_mf, state_idx):
        '''calculate the logits of params (theta_ik) of state in a node Xi, 

        Equation: log theta = E[ xi, x, e]

        args: 
            t_node_mf: i as the index of the node
            state_idx: k as the index of the state in i node 

        output: 
            The logits of the parameter 
        '''
        # build an indicator function of I(xi=k): 
        # one hot distribution [ 0, 0, 1, 0, 0]
        init_params = np.zeros( [ t_node_mf.nstates])
        init_params[ state_idx] = 1.
        t_node_mf.cpt = factor( [t_node_mf.name], init_params)
        
        # calculate the E q(xi, x-i, e) [ log p( xi, x-i, e)]
        # as q(xi)
        log_sum = 0.
        for node_name in self.graph_mf.name_nodes:
            t_node_true = self.graph.get_node( node_name)

            # build q(xl, pi(xl))
            prob = self.graph_mf.get_node( node_name).cpt
            sum_over = [node_name]
            if len(t_node_true.parents): 
                parents_name = [ par_node.name for par_node in t_node_true.parents]
                prob = fac_prod( prob, self.joint_parents( parents_name))
                sum_over += parents_name

            # sum_xl, q(xl, pi(xl)) log p( xl|pi(xl))
            prob = fac_sum( fac_prod( prob, t_node_true.cpt.log()), sum_over)

            # add to logits 
            log_sum += prob.get_distribution()
            
        return log_sum

    def joint_parents( self, parents_name):
        '''Calculate q(Z1,Z2,...,Zk), Zk in Pi(Zk)

        Toh help calculate the expectation term. 
        q(Z1,Z2,...,Zk) = \prod q(Z1)q(Z2)...q(Zk)

        args: 
            name of the parents: We can use the node name to index
            the node, then q
        '''
        for n, node_name in enumerate(parents_name):
            if n==0:
                prob = self.graph_mf.get_node( node_name).cpt
            else:
                prob = fac_prod( prob, self.graph_mf.get_node( node_name).cpt)
        return prob

    def show_convergence( self, deltas, query):
        plt.style.use( 'ggplot')
        plt.figure( figsize = [ 8, 6])
        x = range( 1, len( deltas) + 1)
        plt.plot( x, deltas, linewidth =2.)
        plt.xlabel( 'iterations')
        plt.ylabel( 'abs change of the parameters')
        plt.title( 'convergence of variational method (p{})'.format(query))
        plt.savefig( 'variation inference {}'.format(query))
      
if __name__ == "__main__":
     
    # load data

    # index: 
    vars_value = OrderedDict({
                'BirthAsphyxia': [ 'Yes', 'No'], 
                'Disease': [ 'PFC', 'TGA', 'Fallot', 'PAIVS', 'TAPVD', 'Lung'],
                'Age': [ '0-3 days', '4-10 days', '11-30 days'], 
                'LVH': [ 'Yes', 'No'], 
                'DuctFlow': [ 'Lt to Rt', 'None', 'Rt to Lt'], 
                'CardiacMixing': [ 'None', 'Mild', 'Complete', 'Transparent'], 
                'LungParench': [ 'Normal', 'Oedema', 'Abnormal'],
                'LungFlow': [ 'Normal', 'Low', 'High'],
                'Sick': [ 'Yes', 'No'], 
                'HypDistrib': [ 'Equal', 'Unequal'], 
                'HypoxiaInO2': [ 'None', 'Moderate', 'Severe'], 
                'CO2': [ 'Normal', 'Low', 'High'], 
                'ChestXray': [ 'Normal', 'Oligaemic', 'Plethoric', 'Grd_Glass', 'Asy/Patchy'], 
                'Grunting': [ 'Yes', 'No'], 
                'LVHreport': [ 'Yes', 'No'], 
                'LowerBodyO2': [ '<5', '5-12', '12+'], 
                'RUQO2': [ '<5', '5-12', '12+'], 
                'CO2Report': [ '<7.5', '>=7.5'], 
                'XrayReport': [ 'Normal', 'Oligaemic', 'Plethoric', 'Grd_Glass', 'Asy/Patchy'],  
                'GruntingReport': [ 'Yes', 'No']
                })

    vars_seq = list(vars_value.keys())

    # load the structure info 
    s = loadmat('data/structure.mat')
    dag = s['dag']
    domain_counts = s['domain_counts']

    # load the parameters 
    with open( 'data/parameter.pkl', 'rb') as handle:
        params = pickle.load( handle)

    # ordered the parameters to pair with domain_counts
    ordered_params = OrderedDict()
    for key in vars_seq: 
        ordered_params[key] =  params[key]
    params = ordered_params

    # fix sequence that follows the topological structure for likelihood weighted method
    seq = ['BirthAsphyxia', 
            'Disease',
            'LVH',
            'DuctFlow', 
            'CardiacMixing', 
            'LungParench',
            'LungFlow',
            'Sick',
            'Age',
            'HypDistrib',
            'HypoxiaInO2', 
            'CO2',
            'ChestXray', 
            'Grunting', 
            'LVHreport', 
            'LowerBodyO2', 
            'RUQO2', 
            'CO2Report', 
            'XrayReport',  
            'GruntingReport'] 

    # Build Bayesian network 
    graph = BayesNet()
    ## add nodes
    for key in vars_seq:
        graph.add_nodes( [ key, vars_value[ key]])
    ## add edges
    for node_idx, node_name in enumerate( vars_seq):
        par_indice = np.where( dag[: , node_idx] == 1)[0]
        parents = [ vars_seq[par_idx] for par_idx in par_indice]
        graph.add_edges( [ parents, node_name, params[ node_name]]) 
    #graph.sanity_check()

    # # liklihood weighted sampling
    #sample_lst = [ 100, 1000, 3000, 10000]
    n_samples = 3000
        #print( 'Sample size = {}'.format( n_samples))
        #for _ in range(3):
    graph.reset()
    sampler = likelihood_weighted_sampler( graph, seq)
    time_start=time.time()
    q1 = sampler.infer( 'BirthAsphyxia', 
                    CO2Report = '<7.5', LVH = 'Yes', XrayReport = 'Plethoric', 
                    mode = 'posterior', n_samples = n_samples)
    time_end=time.time()
    print( "For liklihood weighed sampling: \n \
        p(BirthAsphyxia|CO2Report<7.5, LVH=Yes, XrayReport =Plethoric)={} \n \
        takes time: {}".format(q1, time_end-time_start))

    graph.reset()
    time_start=time.time()
    q2 = sampler.infer( 'Disease', 
                    CO2Report = '<7.5', LVH = 'Yes', XrayReport = 'Plethoric', 
                    mode = 'MAP', n_samples = n_samples)
    time_end=time.time()               
    print( "For liklihood weighed sampling: \n \
        p(Disease|CO2Report<7.5, LVH=Yes, XrayReport =Plethoric)={} \n \
        takes time: {}".format(q2, time_end-time_start))

    # Gibbs sampling
    # burnlst = [ 30, 3000, 10000]
    # skiplst = [ 1, 4, 10, 20]

    # for burnin in burnlst:
    #     for skip_step in skiplst:

            # print( 'burn-in: {}, skiplst: {}'.format( burnin, skip_step))
            # for _ in range( 6):
    burnin = 6000
    skip_step = 4
    n_samples = 10000
    graph.reset()
    sampler = gibbs_sampler( graph, burnin = burnin, skip_step=skip_step)
    time_start=time.time()
    q1 = sampler.infer( 'BirthAsphyxia', 
                    CO2Report = '<7.5', LVH = 'Yes', XrayReport = 'Plethoric', 
                    mode = 'posterior', n_samples = n_samples)
    time_end=time.time()
    print( "For Gibbs sampling: \n \
        p(BirthAsphyxia|CO2Report<7.5, LVH=Yes, XrayReport =Plethoric)={} \n \
        takes time: {}".format(q1, time_end-time_start))

    time_start=time.time()
    q2 = sampler.infer( 'Disease', 
                    CO2Report = '<7.5', LVH = 'Yes', XrayReport = 'Plethoric', 
                    mode = 'MAP', n_samples = n_samples)
    time_end=time.time()               
    print( "For Gibbs sampling: \n \
        p(Disease|CO2Report<7.5, LVH=Yes, XrayReport =Plethoric)={} \n \
        takes time: {}".format(q2, time_end-time_start))

     # Variational method
    graph.reset()
    approximator = variation_method( graph, verbose=True, show_plot=False)
    time_start=time.time()
    q1 = approximator.infer( 'BirthAsphyxia', 
                    CO2Report = '<7.5', LVH = 'Yes', XrayReport = 'Plethoric', 
                    mode = 'posterior')
    time_end=time.time()
    print( "For variational method: \n \
        p(BirthAsphyxia|CO2Report<7.5, LVH=Yes, XrayReport =Plethoric)={} \n \
        takes time: {}".format(q1, time_end-time_start))

    time_start=time.time()
    approximator.reset()
    q2 = approximator.infer( 'Disease', 
                    CO2Report = '<7.5', LVH = 'Yes', XrayReport = 'Plethoric', 
                    mode = 'MAP')
    time_end=time.time()               
    print( "For variational method: \n \
        p(Disease|CO2Report<7.5, LVH=Yes, XrayReport =Plethoric)={} \n \
        takes time: {}".format(q2, time_end-time_start))


    #############################  Notes For grader #####################################
    #  
    # 1. Run the script and the results will be shown in the terminal window
    # 2. I have to admit that my gibbs sampling does not work. It is not stable 
    #    and I cannot find the problem  
    #
    ###################################################################################


    
        
            





