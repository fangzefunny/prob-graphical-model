import numpy as np 
from utils.prob_tool import *

class belief_prop:
    
    def __init__( self, graph, tol = 1e-3, max_epi = 10, verbose=False, show_plot=False):
        self.graph = graph
        self.sequence = []
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

    def reset( self):
        self.graph.reset()
        self.sequence = []
            
    def init_belief_table( self):
        '''
        init belief table, prepare for the output 
        '''
        self.belief_table = {}
        for node_name in self.sequence:
            i_node = self.graph.get_node( node_name)
            self.belief_table[ node_name] = np.zeros( [ i_node.nstates,])    
    
    def fix_evidence( self, evid_str):
        '''
        input the evidence, and initiate the cached pi_msg and lambda_msg
        '''
        evid_str = np.array(list(evid_str))
        eq_idx = np.where( evid_str == '=')[0]
        evid_vars = evid_str[eq_idx-1]
        evid_states = evid_str[eq_idx+1]
        for evid_name, evid_state in zip( evid_vars, evid_states):
            evid_node = self.graph.get_node( evid_name)
            evid_node.register_evid( evid_state)
            evid_node.prior_msg = evid_node.curr_dist
            evid_node.likelihood_msg = evid_node.curr_dist

        node_list = self.graph.name_nodes.copy()
        for evid_name in evid_vars:
            node_list.pop( node_list.index(evid_name))
        
        if len(self.sequence) == 0:
            self.sequence = np.random.permutation( node_list)

        for node_name in node_list:
            i_node = self.graph.get_node( node_name)
            if len( i_node.parents) == 0:
                i_node.prior_msg = i_node.cpt
                i_node.likelihood_msg = factor( [node_name], np.ones( [i_node.nstates,]))
            elif len( i_node.children) == 0:
                i_node.likelihood_msg = factor( [node_name], np.ones( [i_node.nstates,]))
                i_node.prior_msg = factor( [node_name], np.ones( [i_node.nstates,]))
            else:
                i_node.likelihood_msg = factor( [node_name], np.ones( [i_node.nstates,]))
                i_node.prior_msg = factor( [node_name], np.ones( [i_node.nstates,])) 
                
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
        tot_likelihood_msg = factor( [ parent.name], np.ones([parent.nstates,]))
        if len( children_no_target):
            for child in children_no_target:
                tot_likelihood_msg = fac_prod( tot_likelihood_msg, 
                                              self.likelihood_msg_from_child( child, parent))
        return fac_prod( parent_prior_msg, tot_likelihood_msg)
    
    def tot_likelihood_msg( self, query):
        '''
        Calculate: Lambda(Xq)
        '''
        tot_likelihood_msg = factor( [ query.name], np.ones([query.nstates,]))
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

        if (len(self.sequence) == 0):
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
            if (wait >= 1) or (epi >= self.max_epi):
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

class direct_method:

    def __init__(self, graph):
        self.graph = graph
    
    def reset(self):
        self.graph.reset()

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
        
        taker_over = self.fix_evidence( evidence )

        # joint distribution p(x,e)
        for n, i_node_name in self.graph.name_nodes:
            i_node = self.get_node( i_node_name)
            if n == 0:
                prob = i_node.cpt
            else:
                prob = fac_prod( prob, i_node.cpt)

        # output 
        if mode == 'sum-product':
            return self.belief_table
        if mode == 'max-product':
            MPE = {}
            for key in self.belief_table:
                MPE[key] = np.argmax( self.belief_table[key])
            return MPE

    def fix_evidence( self, evid_str):
        '''
        input the evidence, and initiate the cached pi_msg and lambda_msg
        '''
        evid_str = np.array(list(evid_str))
        eq_idx = np.where( evid_str == '=')[0]
        evid_vars = evid_str[eq_idx-1]
        evid_states = evid_str[eq_idx+1]
        take_over = []
        for evid_name, evid_state in zip( evid_vars, evid_states):
            evid_node = self.graph.get_node( evid_name)
            evid_idx = evid_node._state_to_idx( evid_state)
            take_over.append( [evid_name, evid_idx])
        return take_over
                

        

