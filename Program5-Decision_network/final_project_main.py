'''
Final project: study ID and DID

'''

import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from collections import namedtuple
from utils import node, bayesNet, factor, fac_prod, fac_max, fac_sum, fac_div, inference, prob_matrix, inference

# find the location of the current file
dir = os.path.dirname(os.path.abspath(__file__))
name_node = namedtuple( 'node', ( 'name', 'content', 'type'))

'''
PART0: define the class of influence diagram that
       An underlying structure for the following algorithm 
'''

class InfluenceDiagrams(bayesNet):
    '''Defines a ID graph.

    Modified based on my own implementation of 
    Bayesian network in project3. 
    '''

    def __init__(self):
        super( InfluenceDiagrams, self).__init__()
        self.decisions = []
        self.utilities = []
        self.nodes = []
        self.edges = [] 
        self.params = []
        self.left_nodes = []

    def add_nodes( self, *args):
        '''Add nodes
        '''
        for item in args:
            node_name, value_idx, node_type = item
            self.nodes += [ name_node( node_name, node( node_name, value_idx, node_type), node_type)]
            if node_type == 'decision':
                self.decisions.append( node_name)
            elif node_type == 'utility':
                self.utilities.append( node_name)

        # an axulliary list to help indexing
        self.name_nodes = list(name_node( *zip( *self.nodes)).name)

    def remove_nodes( self, node_name):
        '''Remove the node from the graph

        Set is_ignore as 1.
        And remove the node from its parents, add its child to its parent
        Remove the node from its children, add parents to its child
        '''
        i_node = self.get_node( node_name)
        i_node.is_ignore = 1.
        if len(i_node.children):
            i_node.children[0].parents.pop( i_node.children[0].parents.index(i_node))
            if len(i_node.parents):
                for par_node in i_node.parents:
                    par_node.children.pop( par_node.children.index(i_node))
                    if i_node.children[0] not in par_node.children:
                        par_node.children.append( i_node.children[0])
                    if par_node not in i_node.children[0].parents:
                        i_node.children[0].parents.append( par_node)
        else:
            if len(i_node.parents):
                for par_node in i_node.parents:
                    par_node.children.pop( par_node.children.index(i_node))
        i_node.children = []
        i_node.parent   = []
                    

    def add_edges( self, *args, mode='regular' ): 
        '''Add links

        'regular' is for paremeterized inforamtion link
        'one-shot' is for non-parameterized information link
        '''
        for item in args:
            from_, to_ = item
            parent = self.get_node( from_)
            child  = self.get_node( to_) 
            if mode =='one-shot':
                if child.node_type != 'utility':
                    parent.children.append( child)
            else:
                parent.children.append( child)
            child.parents.append( parent)
    
    def init_params( self, params_init = 'uniform'):
        for i_node in self.nodes:
            if i_node.type != 'decision':
                i_node.content._init_cpt( params_init)
        self.get_par_config()
    
    def load_params( self, params, mode='regular'):
        '''From 2D params to factor cpt

        Input:
            params: a dictionary contains, a 2D params dict
            e.g for a 2D param dist
            p_AB = { 'B=0': [ .5, .5],
                     'B=1': [ .4, .6]} 

        'regular' is for paremeterized inforamtion link
        'non-param-info-link' is for non-parameterized information link            
        '''
        for i_node_name in self.name_nodes:
            i_node = self.get_node( i_node_name)
            if mode == 'non-param-info-link':
                if i_node.node_type == 'decision':
                    pass
                else:
                    i_params = params[ i_node_name]
                    i_node._params_to_cpt( i_params)
            else: 
                i_params = params[ i_node_name]
                i_node._params_to_cpt( i_params)

    def vis_params( self):
        '''
        Check if the code is bug-free 
        '''
        for node_name in self.name_nodes:
            i_node = self.get_node( node_name)
            if i_node.is_ignore == 0:
                print( 'For {}, type: {}, parents: {}, children: {} \n CPT:'.format( 
                            i_node.name,
                            i_node.node_type,  
                            str([par_node.name for par_node in i_node.parents]), 
                            str([child_node.name for child_node in i_node.children])))
                if i_node.cpt:
                    print(i_node.cpt.show_dist())


'''
PART1: Implement of Algorithm 1. 
        A extremely native method only works in one-shot case.
        This method can evaluate the policy but not
        able to calculate the MEU. 
'''

class oneShotID:

    def __init__( self, id):
        self.network = id
        self.infer = inference.belief_prop( self.network)
    
    def evaluate( self, evid = {}):
        '''Implement of alg 1

        A extremely native method only works in one-shot case.
        This method can evaluate the policy but not
        able to calculate the MEU. 
        '''
        # due to one shot algorithm, we can get the first
        # value in the decisions sequence
        decision = self.network.get_node( self.network.decisions[0])
        util_node = self.network.get_node(self.network.utilities[0])
        # If there is an inforamtion link to the decision
        if len(decision.parents):
            out_dict = {}
            par_node = decision.parents[0]
            # remove the information link to avoid inputing dependency.
            self.network.remove_edges( [ par_node.name, decision.name ])
            # use loop to fix evidence for observed node and 
            for state in par_node.states:
                temp1_evid = evid.copy()
                temp1_evid[ par_node.name] = state
                Eutil_lst = []
                par_config = '{}={}'.format(par_node.name, state)
                for act in decision.states:
                    temp_evid = temp1_evid.copy()
                    temp_evid[self.network.decisions[0]] = act
                    #clear the belief table
                    self.infer.reset()
                    belief = self.infer.inference( evidence = temp_evid, utility=util_node.name,
                                                mode = 'sum-product')

                    Eutil = util_node.cpt 
                    for i_par_node in util_node.parents:
                        i_par_name = i_par_node.name
                        try:
                            i_par_prob = belief[ i_par_name]
                        except:
                            i_par_prob = np.array(self.network.get_node(i_par_name
                                                    ).curr_dist.get_distribution())
                        Eutil = fac_sum( fac_prod( Eutil, factor( [i_par_name], 
                                        i_par_prob)), [i_par_name])
                    Eutil_lst.append( Eutil.get_distribution())
                    results = { 'Expected utility': np.round(Eutil_lst.copy(),4), 
                                        'Decision': decision.states[np.argmax(Eutil_lst)]}
                    out_dict[ par_config] = results

            return out_dict
        else:
            out_dict = {}
            results = {}
            Eutil_lst = []
            for act in decision.states:
                temp_evid = evid.copy()
                temp_evid[self.network.decisions[0]] = act

                #clear the belief table
                self.infer.reset()
                belief = self.infer.inference( evidence = temp_evid, utility=util_node.name,
                                            mode = 'sum-product')

                Eutil = util_node.cpt 
                for par_node in util_node.parents:
                    par_name = par_node.name 
                    Eutil = fac_sum( fac_prod( Eutil, factor( [par_name], 
                                    belief[ par_name])), [par_name])
                Eutil_lst.append( Eutil.get_distribution())
                results = { 'Expected utility': np.round(Eutil_lst.copy(),4), 
                            'Decision': decision.states[np.argmax(Eutil_lst)]}
                out_dict[ 'unknown_term'] = results         
            return out_dict


'''
PART2: Implement of Algorithm 2. 
        Exact method introduced in Shachter86.
        However, my implmenetation is based on the revised version
        in Shachter2007, because it is more readable.
'''

class direct_method:

    def __init__( self, id):
        self.network = id
        self.infer = inference.belief_prop( self.network)

    def evaluate( self, evid = {}):
        '''Implement of alg 2

        Where i implement the pseudo-code
        '''
        # to store the optimal solutions
        optimal_decisions = []
        # insert any evidence
        for node_name in evid.keys():
            i_node = self.network.get_node( node_name)
            i_node.register_evid(evid[node_name])
            i_node.cpt = i_node.curr_dist
            if i_node.node_type == 'decision':
                optimal_decisions.append(i_node.curr_state)

        # check those nodes waited to be eliminated,
        # neither evidence nor the value node
        self.check_left_node()

        while len(self.network.left_nodes) > 1: 
            #self.network.vis_params()
            # remove a barren node if possible 
            remove_or_not = self.remove_barren_node()
            if remove_or_not==False:
                # remove a decision node if possible 
                remove_or_not, decision = self.remove_decision_node()
                if len(decision): 
                    optimal_decisions.insert(0, decision) # record the policy 
                if remove_or_not==False:
                    # remove a chance node if possible 
                    remove_or_not = self.remove_chance_node()
                    if remove_or_not==False:
                        # if there is no removable node, reverse arc
                        remove_or_not = self.reverse_arc()
                        if remove_or_not==False:
                            remove_or_not = self.remove_value_node()
        #self.network.vis_params()
        EU = self.network.get_node(self.network.left_nodes[0]).cpt.get_distribution()
        return EU, optimal_decisions

    def check_left_node(self):
        self.network.let_nodes = [] 
        for node_name in self.network.name_nodes:
            i_node = self.network.get_node( node_name)
            if i_node.is_evid * i_node.is_ignore==0:
                self.network.left_nodes.append(node_name)

    def remove_barren_node(self):
        for node_name in self.network.left_nodes:
            i_node = self.network.get_node(node_name)
            if i_node.node_type == 'utility':
                pass
            else: 
                if len(i_node.children)==0:
                    self.network.remove_nodes( node_name)
                    self.network.left_nodes.pop( 
                        self.network.left_nodes.index(node_name))
                    print('remove barren node', node_name)
                    return True 
        return False 
    
    def remove_decision_node( self):
        '''Remove a chance node

        implementation of equation 2
        '''
        for node_name in self.network.left_nodes:
            i_node = self.network.get_node(node_name)
            if i_node.node_type == 'decision':
                if len(i_node.children)==1:
                    child_node = i_node.children[0]
                    node_par_set = set(i_node.parents)
                    if child_node.node_type == 'utility':
                        if (set(child_node.parents) in node_par_set.union(set([i_node]))) or \
                           (set(child_node.parents) == node_par_set.union(set([i_node]))):
                            new_factor, policy = fac_max( fac_prod( i_node.cpt, child_node.cpt), node_name)
                            child_node.cpt = new_factor
                            self.network.remove_nodes( node_name)
                            self.network.left_nodes.pop( 
                                self.network.left_nodes.index(node_name))
                            print('remove decision node', node_name)
                            policy_idx = dict()
                            if policy.shape[0]> 1: 
                                for i_par_node in i_node.parents:
                                    for n, state in enumerate(i_par_node.states):
                                        fname = '{}={}'.format(i_par_node.name, state)
                                        policy_idx[fname] = policy[n]
                                    return True, pd.DataFrame(policy_idx, index=['decisions'])
                            else: 
                                True, policy     
        return False, [] 

    def remove_chance_node( self):
        '''Remove a chance node

        implementation of equation 3
        '''
        for node_name in self.network.left_nodes:
            i_node = self.network.get_node(node_name)
            if i_node.node_type == 'chance':
                if len(i_node.children)==1:
                    child_node = i_node.children[0]
                    if child_node.node_type == 'utility':
                        new_factor = fac_sum( fac_prod(child_node.cpt, i_node.cpt), node_name)
                        child_node.cpt = new_factor
                        self.network.remove_nodes( node_name)
                        self.network.left_nodes.pop( 
                                self.network.left_nodes.index(node_name))
                        print('remove chance node', node_name)
                        return True
        return False

    def check_reversible( self, i_node_name, j_node_name):
        '''Check if the link is reversible 

        implementation of equation 5
        '''
        # A recurisve function
        def visit_yet( i_node, j_node, visited_lst):
            # if the "new node" has been visted 
            if j_node in visited_lst:
                return True
            # record the visited node
            visited_lst.append( i_node)
            # follow the children
            if len(i_node.children):
                for child_node in i_node.children:
                    if visit_yet(child_node, j_node, list(visited_lst)):
                        return True 
            return False
        # empty list before visit
        i_node = self.network.get_node( i_node_name)
        j_node = self.network.get_node( j_node_name)
        if visit_yet( i_node, j_node, list()):
            return True
        return False

    def reverse_arc( self):
        '''Reverse a link, application of bayes theroem
        This is not learned in the class, so it is hard 
        to implment this
        '''
        for node_name in self.network.left_nodes:
            i_node = self.network.get_node(node_name)
            if i_node.node_type == 'chance':
                for j_node_name in self.network.left_nodes:
                    if node_name != j_node_name:
                        j_node = self.network.get_node(j_node_name)
                        if j_node.node_type == 'chance':
                            self.network.remove_edges( [node_name, j_node_name])
                            loopy = self.check_reversible( node_name, j_node_name)
                            if 1 - loopy: 
                                both_parents  = set(j_node.parents).intersection(set(i_node.parents))
                                new_i_parents = set(j_node.parents).union(set([j_node])) - (both_parents)
                                new_j_parents = set(i_node.parents) - (both_parents)
                                if len(new_i_parents):
                                    for i_par_node in new_i_parents:
                                        self.network.add_edges( [i_par_node.name, node_name])
                                if len(new_j_parents):
                                    for j_par_node in new_i_parents:
                                        self.network.add_edges( [j_par_node.name, j_node_name])
                                joint = fac_prod( i_node.cpt, j_node.cpt)
                                new_j_factor = fac_sum( joint, [node_name])
                                new_i_factor = fac_div( joint, new_j_factor)
                                i_node.cpt = new_i_factor 
                                j_node.cpt = new_j_factor
                                print( 'revser arc from {} to {}'.format(node_name, j_node_name))
                                return True
                            else:
                                self.network.add_edges( [node_name, j_node_name])
                        
        return False

    def remove_value_node( self):
        '''Combine a value node 

        Implement of equation 4.
        '''
        for node_name in self.network.left_nodes:
            i_node = self.network.get_node(node_name)
            if i_node.node_type == 'value':
                for j_node_name in self.network.left_nodes:
                    if node_name != j_node_name:
                        j_node = self.network.get_node(j_node_name)
                        if j_node.node_type == 'value':
                            i_node.cpt = factor( [i_node.get_variables()], 
                                            i_node.get_distribution() + j_node.get_distribution())
                            print( 'revser arc from {} to {}'.format(node_name, j_node_name))
                            return True
        return False

    
'''
Implement a simple MDP game that allow fix length decision process
'''

class survial:

    def __init__(self, max_t = 7):
        self.p0 = [ 1, 0, 0]
        self.t = 0
        self.max_t = max_t  
        self.done = False 
        self.state_dim = 3 
        self.action_dim = 2 
        self.state_config = [ 'high', 'low', 'exhausted']
        self.action_config = [ 'search', 'wait']
        p1 = .4
        p2 = .6
        p3 = .7
        p4 = .8
        r1 = 8.   # search with high energy  
        r2 = 4.   # search with low energy
        r3 = -50. # search with zero energy
        r4 = -4.  # wait with high energy
        r5 = -2.  # wait with low energy
        r6 = -1.  # wait when tired  
        self.transition = { 'high, search':       [ p1, 1 - p1, 0],
                            'low, search':        [ 0, p2, 1 - p2],
                            'exhausted, search':  [ 0, 0, 1],
                            'high, wait':         [ 1, 0, 0], 
                            'low, wait':          [ p3, 1-p3, 0],
                            'exhausted, wait':    [ 0, p4, 1 - p4]}
        self.reward = { 'high, search':     [ r1],
                        'low, search':      [ r2],
                        'exhausted, search':[ r3],
                        'high, wait':       [ r4], 
                        'low, wait':        [ r5],
                        'exhausted, wait':  [ r6]} 
        self._for_sample()
    
    def _for_sample( self):
        self.transition_for_sample = dict()
        self.reward_for_sample = np.zeros([ self.state_dim, self.action_dim])
        for state in range(self.state_dim):
            self.transition_for_sample[state] = dict()
        for n, tkey, rkey in zip(range(self.state_dim * self.action_dim), 
                                self.transition.keys(), self.reward.keys()):
            state = n % self.state_dim
            action = n // self.state_dim
            self.transition_for_sample[state][action] = self.transition[tkey]
            self.reward_for_sample[ state, action ] = self.reward[rkey][0]

    def reset( self):
        '''Init the MDPs

        start at t=0
        '''
        self.t = 0 
        self.done = False 
        self.state = np.random.choice(range(self.state_dim), p=self.p0)
        return self.state 

    def step( self, action):
        '''Make a step in MDP

        develop mimic the gym environment
        '''
        self.t += 1 
        prob = self.transition_for_sample[self.state][action]
        self.state = np.random.choice(range(self.state_dim), p=prob)
        reward = self.reward_for_sample[self.state, action]
        if self.t >= self.max_t:
            self.done = True 
        return self.state, reward, self.done

    def generate_data( self, sample_size = 100):
        '''Generate data for learning 
        '''
        trajectories = []
        for _ in range(sample_size):
            done = False 
            state = self.reset()
            one_traj = []
            while not done: 
                action = np.random.choice(range(self.action_dim))
                next_state, reward, done = self.step(action)
                one_traj += [state, action, reward]
                state = next_state
            trajectories.append( one_traj)
        return trajectories

    def show_params( self):
        return pd.DataFrame( self.transition), pd.DataFrame( self.reward, index=['rewards'])

'''
PART3:  Implement of Algorithm 3. 
        Backward algorithm that is the same as the algorithm2
        require the fixed length MDP
'''
class backward_recursive:

    def __init__( self, mdp):
        self.mdp = mdp
        self.mdp_length = self.mdp.max_t
        self.state_dim = self.mdp.state_dim
        self.action_dim = self.mdp.action_dim 
        self.Expected_utility = {}
        self.policies = {}

    def evaluate( self):
        '''Evaluate MDP using recusive method
        '''
        # obtain parameter from the evironemnt
        reward = prob_matrix(self.mdp.reward, [self.state_dim, self.action_dim])
        transition =  prob_matrix(self.mdp.transition, 
                      [self.state_dim, self.state_dim, self.action_dim])
        # init Q^T(s,a) = R(s,a)
        Qtable = reward
        
        # start from the last second step:
        for t in reversed(range(self.mdp_length)):
            
            # store the value using dynamic programming idea
            policy, maxQ = np.argmax(Qtable, axis=1), np.max(Qtable, axis=1)
            self.policies[t] = policy 
            self.Expected_utility[t] = maxQ 
             
            # Q(s,a) = r(s,a) + sum_{s'} p(s'|s,a)max_{a'}Q(s',a')
            if t==0:
                MEU = np.sum(self.mdp.p0 * maxQ)
                return MEU, self.policies, self.Expected_utility
            else:
                Qtable = reward.copy()
                for i in range(self.state_dim):
                    Qtable += transition[ i, :, :] * maxQ[i] 

'''
PART4:  Implement of Algorithm 4. 
        Revised version of algorithm 3 inpisred by RL and belief propagation
        does not require the fixed length MDP
        does not require full acknowledge of previous decisions
'''
class forward_recursive:

    def __init__( self, mdp):
        self.mdp = mdp
        self.mdp_length = self.mdp.max_t
        self.state_dim = self.mdp.state_dim
        self.action_dim = self.mdp.action_dim 
        self.Expected_utility = {}
        self.policies = {}
        self._init_est()
    
    def _init_est( self):
        # guess the value at each state 
        for t in range(self.mdp_length+1):
            self.Expected_utility[t] = np.zeros([ self.state_dim,])

    def evaluate( self):
        '''Evaluate MDP using recusive method
        
        '''
        # obtain parameter from the evironemnt
        reward = prob_matrix(self.mdp.reward, [self.state_dim, self.action_dim])
        # next_state, current_state, action 
        transition =  prob_matrix(self.mdp.transition, 
                      [self.state_dim, self.state_dim, self.action_dim])

        done = False 
        epi  = 0
        deltas = []
        # start from the last second step:
        while not done: 

            old_MEU = np.sum(self.mdp.p0 * self.Expected_utility[0])
            for t in range(self.mdp_length):
                     
                # r(s,a) + sum p(s'|s,a)max_{a'}Q(s',a')
                if t==self.mdp_length-1:
                    print('1')
                maxQ = self.Expected_utility[t+1]

                Qtable = reward.copy()
                for i in range(self.state_dim):
                    Qtable += transition[ i, :, :] * maxQ[i] 

                # store the value as update
                policy, maxQ = np.argmax(Qtable, axis=1), np.max(Qtable, axis=1)
                self.policies[t] = policy 
                self.Expected_utility[t] = maxQ 
    
            new_MEU = np.sum(self.mdp.p0 * self.Expected_utility[0])
            delta = new_MEU - old_MEU
            deltas.append(delta)
            print( 'Epi: {}, change: {}'.format( epi, delta))
            epi += 1 
            if delta < 1e-3:
                done = True 
                return new_MEU, self.policies, self.Expected_utility, deltas

'''
PART5:  Learning the paramter in MDP,
        use the count method
'''
class MDP_learning:

    def __init__(self, mdp):
        self.mdp = mdp
        self.state_dim = self.mdp.state_dim
        self.action_dim = self.mdp.action_dim
        self._init_trans_func()

    def _init_trans_func(self):
        self.transition = np.zeros( [self.state_dim, self.state_dim, self.action_dim])
        self.reward     = np.zeros( [self.state_dim, self.action_dim])

    def learn( self, data):
        # slipt the data into state trajectory, action trajectory, reward trajectory
        for trajectory in data:
            states  = []
            actions = []
            rewards = []
            for n, i in enumerate(trajectory):
                if n % 3 ==0:
                    states.append(i)
                elif n % 3==1:
                    actions.append(i)
                else:
                    rewards.append(i)
            
            # learn transition
            for t in range(len(states)-1):
                self.transition[ states[t+1], states[t], actions[t]] += 1
                if rewards[t] != self.reward[ states[t], actions[t]]:
                    self.reward[ states[t], actions[t]] = rewards[t]
            
        # noramlize transition function
        transition = self.transition / np.sum( self.transition, axis=0, keepdims=True)
        reward = self.reward

        self.mdp.transition = dict()
        self.mdp.reward     = dict()
        for n in range(self.state_dim*self.action_dim):
            state = n % self.state_dim
            action = n // self.state_dim
            fname = '{}, {}'.format( self.mdp.state_config[state], 
                                    self.mdp.action_config[action])
            self.mdp.transition[fname] = transition[:, state, action]
            self.mdp.reward[fname] = [reward[state,action]]


# load the fever example 
def load_fever_exp( mode = 'regular'):
    # handmade cpt
    params = {'flu': { 'prior': [ .05, .95]}, 
              'fever': { 'flu=True' : [ .95, .05],
                         'flu=False': [ .02, .98]},
              'therm': { 'fever=True' : [ .90, .10],
                        'fever=False': [ .05, .95]},
              'takeAspirin': { 'therm=True' : [ 1, 1],
                               'therm=False': [ 1, 1]},
              'feverLater': { 'fever=True, takeAp=Yes': [ .05, .95],
                              'fever=False, takeAp=Yes':[ .01, .99],
                              'fever=True, takeAp=No':  [ .90, .10],
                              'fever=False, takeAp=No': [ .02, .98]}, 
              'reaction': { 'takeAp=Yes': [ .05, .95],
                            'takeAp=No':  [   0,   1]}, 
              'utility': { 'feverLa=True, reaction=Yes':  [-50],
                           'feverLa=False, reaction=Yes': [-10],
                           'feverLa=True, reaction=No':   [-30],
                           'feverLa=False, reaction=No':  [ 50]}}

    network = InfluenceDiagrams()
    network.add_nodes(
        [ 'flu',   [ 'True', 'False'], 'chance'],
        [ 'fever', [ 'True', 'False'], 'chance'],
        [ 'therm', [ 'True', 'False'], 'chance'],
        [ 'takeAspirin', [ 'Yes', 'No'], 'decision'],
        [ 'feverLater', [ 'True', 'False'], 'chance'],
        [ 'reaction', [ 'Yes', 'No'], 'chance'],
        [ 'utility', ['score'], 'utility'],
    )
    if mode == 'regular':
        network.add_edges(
            [ 'flu', 'fever'],
            [ 'fever', 'therm'],
            [ 'therm', 'takeAspirin'],
            [ 'fever', 'feverLater'],
            [ 'takeAspirin', 'feverLater'],
            [ 'takeAspirin', 'reaction'],
            [ 'reaction', 'utility'],
            [ 'feverLater', 'utility'],
        )
    else: 
        network.add_edges( 
            [ 'flu', 'fever'],
            [ 'fever', 'therm'],
            [ 'therm', 'takeAspirin'],
            [ 'fever', 'feverLater'],
            [ 'takeAspirin', 'feverLater'],
            [ 'takeAspirin', 'reaction'],
            [ 'reaction', 'utility'],
            [ 'feverLater', 'utility'],
            mode ='one-shot',
        )
    network.load_params( params, mode=mode)
    return network


def plot_convergence( deltas):
    plt.figure(figsize=(8,6))
    plt.style.use( 'seaborn-poster')
    plt.plot(  deltas, 
                'o-', linewidth=3)
    plt.xlabel('iterations')
    plt.ylabel('convergence condition')
    plt.title( 'Convergence condition with forward')
    plt.savefig( dir+ 'converge_condi.png')
    plt.close()

if __name__ == "__main__":

    # ################################################
    # #####     Exp 1: one-shot ID evaluation    #####
    # ################################################

    # without information link: algorithm1 
    network = load_fever_exp(mode='non-param-info-link')
    eval_alg = oneShotID(network)
    print( pd.DataFrame(eval_alg.evaluate()))

    # use automatic: algorithm2
    network = load_fever_exp(mode='regular')
    eval_alg = direct_method(network)
    evid = { 'therm': 'True'}
    MEU, policy = eval_alg.evaluate(evid)
    print( 'MEU: ', MEU)
    print( 'The policy is:', policy)
    
    #############################################################
    #####        Exp 2: Dynamic ID with surival game        #####
    #############################################################

    ## exp with algorithm3
    mdp = survial(max_t=7)
    alg = backward_recursive( mdp)
    MEU, policies, eus= alg.evaluate()
    print(MEU)
    print('early: {} late: {}'.format(policies[0], policies[6]))

    ## exp with algorithm4
    mdp = survial(max_t=7)
    alg = forward_recursive( mdp)
    MEU, policies, eus, deltas= alg.evaluate()
    print(MEU)
    print('early: {} late: {}'.format(policies[0], policies[6]))
    plot_convergence(deltas)

    #########################################################
    #####        Exp 3: Learning in dynamic ID          #####
    #########################################################

    mdp = survial(max_t=7)
    data = mdp.generate_data(100)
    mdp2 = survial(max_t=7)
    mdp2.transition = 0
    mdp2.reward = 0 
    learner = MDP_learning(mdp2).learn(data)
    print(mdp2.show_params())