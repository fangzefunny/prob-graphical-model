'''
Project4: Image segmentation using MRF

Zeming Fang 
669158922


README:  

Run the script. The whole program takes about 7 min. For the interest of
time, I use a truncated experiment.

You will get a "results" folder at the current folder. Inside
the folder, you should find: 

* A convergence figure: This is the figure3 in my report.
* Many label images: The images generated after each iterations 
  (summarized by figure4, figure5 in my report)
'''
import os
import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt 

# find the location of the current file
dir = os.path.dirname(os.path.abspath(__file__))

'''
PART0: Implementation of potts_model
'''
class potts_model:

    def __init__( self, dim0, dim1):
        self.dim0   = dim0
        self.dim1   = dim1
        self.obs    = np.zeros( [dim0, dim1])
        self.labels = np.zeros( [dim0, dim1])
        # fix paremeters according to the class instruction 
        self.beta        = 2.3472
        self.alpha       = 0.3975
        self.mean0_lab   = 106.2943
        self.mean255_lab = 141.6888
        self.std0_lab    = 5.3705, 
        self.std255_lab  = 92.0635    

    def assign_algs( self, mode='icm'):
        '''Choose the MAP inference algorithm 
        '''
        if mode=='icm':
            self.infer_alg = ICM(self)
        elif mode=='gibbs':
            self.infer_alg = Gibbs(self)

    def init_obs( self, observations):
        '''Init observations

        register the images as evidence. Meanwhile, I calculate
        an axuillary big matrix for self.unary_enegy()
        '''
        self.obs = observations
        self.label0 = -np.log(stats.norm.pdf( self.obs, self.mean0_lab, self.std0_lab))
        self.label255 = -np.log(stats.norm.pdf( self.obs, self.mean255_lab, self.std255_lab))
    
    def init_labels( self, mode='uniform'):
        '''Choose the init method
        '''
        if mode == 'uniform':
            self.labels = np.random.binomial( n=1, p=.5, size=[self.dim0, self.dim1]) * 255
        elif mode == 'zeros':
            self.labels = np.zeros( [self.dim0, self.dim1])
        elif mode == '255s':
            self.labels = np.ones( [self.dim0, self.dim1]) * 255
        else: 
            raise Exception( 'Use the correct init method')
        
    def four_neigbor_idx( self, i_idx):
        '''Find the neighour indinces

        Input:
            i_idx: the location index of the target label

        Output:
            the neighbour location indinces following the 4 neighbour criteria
        '''
        idx = np.array(i_idx)
        neig_lst = []
        directions = [np.array([-1,  0]), # up
                      np.array([ 1,  0]), # down
                      np.array([ 0, -1]), # left
                      np.array([ 0,  1])] # right
        for i_direct in directions:
            neig_idx = idx + i_direct
            if (neig_idx[0] >= 0) and (neig_idx[0] < self.dim0
                ) and (neig_idx[1] >=0) and (neig_idx[1] < self.dim1):
                neig_lst.append( tuple(idx + i_direct))
        return neig_lst 

    def unary_energy( self, i_idx):
        '''Calculate Ei(xi, yi)

        The energy is calculated via the follow equation.
        here we assume 
        
        Ei(xi, yi) = -ln( norm(y| [mu0, mu255], [std0, std255]) ) 

        where mu0=106.2943, std0=5.3705
              mu255=141.6888, std255=92.0635

        Note that, for the interest of the speed, I run a big matrix
        at self.init_obs. In this function I just gather the results
        from that big matrix. 

        Input:
            label_idx: location index of the label, we don't need
            the observation index, because they share the same loc
            idx. 

        Output:
            Ei(xi, yi)
        '''
        # get the observation(yi) and map the observation
        # assign the parameter according to the homework instruction  
        return np.array( [self.label0[ i_idx[0], i_idx[1]], self.label255[ i_idx[0], i_idx[1]]])

    def pair_energy( self, j_idx):
        '''Calculate Eij(xi, xj)

        The energy is calculated via the follow equation.
        The equation is not elegant. because there is
        not i_idx in the input of the function
        Eij(xi,xj) = 1 - delta( xi - xj)

        Input:
            j_idx: location index of j node (xj)

        Output:
            Eij(xi, xj)
        '''
        # get the value of xi and xj
        i_lab_state = [ 0, 255]
        j_lab_state = [ self.labels[ j_idx[0], j_idx[1]]]*2
        return 1 - (np.array(i_lab_state) == np.array(j_lab_state))

    def infer_background( self, observations):
        '''Image segmentation

        Infer what is the foreground and what is the
        background of the input image: 

        Input: 
            observation: input image 

        Ouput:
            execute the inference, return the loss
            history.
        '''
        return self.infer_alg.infer( observations)
'''
PART1: Implementation of ICM 
'''
class ICM:

    def __init__( self, mrf, max_epi=20, 
                             verbose=True,
                             init_method='uniform'):
        self.mrf = mrf
        self.max_epi = max_epi
        self.versboe = verbose
        self.init_method = init_method
    
    def infer( self, observations):
        '''MAP inference using ICM

        This where I implment algorithm 1 to calcuate: 
            x^* = argmax_x p(x|y=observations)

        A confusion I had:
        
        * MAP or MPE: The text book and slide says 
        this is an MAP inference, but I think it is
        a MPE problem. Here I keep the 'MAP' name, think
        about changing it later.

        Input:
            observations: Y, the 2D-image matrix 

        Output:
            most possible explanation: label*
        '''
        # input observations(evidene)
        self.mrf.init_obs( observations)
        # init the unobserved variables (labels)
        self.mrf.init_labels( self.init_method)

        done = False
        epi  = 0
        deltas = []
        while not done: 
            # cache the labels value in the previous iteration
            old_labels = self.mrf.labels.copy()
            for i_row in range(self.mrf.dim0):
                for i_col in range(self.mrf.dim1):
                    # x^t+1 = argmax_xi p(xi|N(xi), y)
                    i_new_state = self.local_MAP( (i_row, i_col))
                    # assign the inferred labels
                    self.mrf.labels[ i_row, i_col] = i_new_state
            # check convergence turns into number of different pixels
            delta = np.sum(abs(self.mrf.labels - old_labels)) / 255
            if (delta == 0) or (epi > self.max_epi):
                done = True
            # record the convergence condition 
            if self.versboe == True:
                print( 'Epi: {}, # of change pixels: {}'.format( epi, delta))
            deltas.append( delta)
            # save the current labels
            self.save_labels( epi)
            epi += 1 
        return deltas

    def local_MAP( self, i_idx):
        '''Caluate the local MAP

        Implment the local MAP inference. However, in reality,
        I am implmenting an equivlant equation:
            x^{t+1} = argmin sum_xj wij Eij(xi,xj) + ai Ei(xi,yi)
        for more detail, see equation 12 and algorithm 1 in my 
        report. 

        Inputs:
            i_idx: the idx of node Xi
        
        output:
            x^t+1: the most possible state og Xi given its
            four neighours
        '''
        # get the indice of four neighor
        neig_indice = self.mrf.four_neigbor_idx( i_idx)
        # cal ai Ei(xi, yi)
        sum_energy = self.mrf.alpha * self.mrf.unary_energy( i_idx)
        # cal sum_xj wij Eij( xi, xj)
        for j_idx in neig_indice:
            sum_energy += self.mrf.beta * self.mrf.pair_energy( j_idx) 
        # argmin equation 12 in project4 report,
        # if argmin pos is 0, return 0
        # if argmin pos is 1, return 255
        new_state = np.argmin( sum_energy) * 255
        return new_state

    def save_labels(self, epi):
        '''Save the label for each iteration
        '''
        plt.figure()
        plt.imshow( self.mrf.labels, cmap='gray')
        plt.axis('off')
        save_name = dir + '/results/icm_{}-init_epi{}.png'.format(
                            self.init_method, epi)
        try: 
            plt.savefig( save_name)
        except: 
            os.mkdir( dir + '/results')
            plt.savefig( save_name)
        plt.close()
'''
PART2: Implementation of Gibbs sampling 
'''
class Gibbs: 

    def __init__( self, mrf, verbose=True,
                             init_method='uniform',
                             burn_in = 10000):
        self.mrf = mrf
        self.versboe = verbose
        self.init_method = init_method
        self.burn_in = burn_in
        self.all_sample_size = 10000
        self.check_interval = self.mrf.dim0 * self.mrf.dim1

    def infer( self, observations):
        '''MAP inference using Gibbs 

        This where I implment algorithm 2 to calcuate: 
            x^* = argmax_x p(x|y=observations)

        Input:
            observations: Y, the 2D-image matrix 

        Output:
            most possible explanation: label*
        '''
        # input observations(evidene)
        self.mrf.init_obs( observations)
        # init the unobserved variables (labels)
        self.mrf.init_labels( self.init_method)

        t = 0 
        burning_in = True
        # start burn-in
        print( '>>>>>> Start burn-in >>>>>>>>')
        while burning_in:
            for i_row in range(self.mrf.dim0):
                for i_col in range(self.mrf.dim1):
                    # sample from pseudo-posterior p()
                    i_new_state = self.sample_pseudo_prob( (i_row, i_col))
                    # assign the inferred labels
                    self.mrf.labels[ i_row, i_col] = i_new_state
                    # check the burn-in 
                    t += 1
                    if self.versboe:
                        if t % self.check_interval == 0:
                            print( 'After {} sweeps....'.format( t / self.check_interval))
                    if t > self.burn_in:
                        burning_in = False
                        break
        
        # start sampling after burn-in
        sum_samples = np.zeros( [ self.mrf.dim0, self.mrf.dim1])
        t = 0 
        done = False 
        print( '>>>>>> Start sampling >>>>>>>>')
        for i_row in range(self.mrf.dim0):
            for i_col in range(self.mrf.dim1):
                # sample from pseudo-posterior p()
                i_new_state = self.sample_pseudo_prob( (i_row, i_col))
                # assign the inferred labels
                self.mrf.labels[ i_row, i_col] = i_new_state
                # collect as sample
                sum_samples += self.mrf.labels
                # check the burn-in 
                t += 1
                if t >= self.all_sample_size:
                    done = True
                    break
            if done:
                break 
    
        # most counts configuration and assign to the mrf
        self.mrf.labels = ((sum_samples / self.all_sample_size) > 255/2 ) * 255
        self.save_labels()

    def sample_pseudo_prob( self, i_idx):
        '''sample from pseudo_prob 

        Implment the local MAP inference. 
            x^{t+1} ~ 1/Z * exp( sum_xj wij Eij(xi,xj) + ai Ei(xi,yi))
        for more detail, see equation 12 and algorithm 2 in my 
        report. 

        Inputs:
            i_idx: the idx of node Xi
        
        output:
            x^t+1: sampled new state for Xi
        '''
        # get the indice of four neighor
        neig_indice = self.mrf.four_neigbor_idx( i_idx)
        # cal ai Ei(xi, yi)
        sum_energy = self.mrf.alpha * self.mrf.unary_energy( i_idx)
        # cal sum_xj wij Eij( xi, xj)
        for j_idx in neig_indice:
            sum_energy += self.mrf.beta * self.mrf.pair_energy( j_idx)
        # build the p(xi|N(xi),y) = 1/Z exp( Energy)
        prob = np.exp( -sum_energy) / np.sum( np.exp( -sum_energy))
        # sample a new_state 
        new_state = np.random.choice( [0,255], p = prob)
        return new_state 

    def save_labels( self):
        '''Save the infered results

        Save the inferred labels after each iteration.
        '''
        plt.figure()
        plt.imshow( self.mrf.labels, cmap='gray')
        plt.axis('off')
        save_name = dir + '/results/gibbs-init_epi{}-burn_in{}.png'.format(
                            self.init_method, self.burn_in)
        try: 
            plt.savefig( save_name)
        except: 
            os.mkdir( dir + '/results')
            plt.savefig( save_name)
        plt.close()


def plot_convergence( deltas, init_method):
    plt.figure(figsize=(8,6))
    plt.style.use( 'seaborn-poster')
    for delta in deltas:
        plt.plot( range(0,len(delta)), delta, 
                  'o-', linewidth=3)
    plt.xlabel('iterations')
    plt.ylabel('num of different pixels')
    plt.ylim([0, 500])
    plt.legend( init_method)
    plt.title( 'Convergence condition with ICM')
    try: 
        plt.savefig( dir + '/results/converge_condi.png')
    except: 
        os.mkdir( dir + '/results')
        plt.savefig( dir + '/results/converge_condi.png')
    plt.close()
        
if __name__ == "__main__":

    ############################################
    ####             Read Image             ####
    ############################################

    image = plt.imread( dir + '/Proj4_image.png') * 255
    dim0 = image.shape[0]
    dim1 = image.shape[1] 

    ########################################################
    ####       exp1: Image Segmentation using ICM       ####
    ########################################################

    # buil a potts model, choose the infer algorithm

    # different init methods
    # 'zeros': all pixels are init to be 0
    # 'uniform': all pxiels are randomly sampled from a uniform beurnoulli distribution
    init_methods = [ 'zeros', 'uniform']
    max_epi = 25 # For grader: set 30 to see the converged results. 
    
    deltas = []
    for method in init_methods:
        # buil a potts model, choose the infer algorithm
        graph = potts_model( dim0, dim1)
        graph.infer_alg = ICM( graph, 
                               init_method=method,
                               max_epi=max_epi)
        c1 = graph.infer_background( image)
        deltas.append(c1)
    
    plot_convergence( deltas, init_methods)

    ########################################################
    ####     exp2: Image Segmentation using Gibbs       ####
    ########################################################

    # different init methods
    # 'zeros': all pixels are init to be 0
    # 'uniform': all pxiels are randomly sampled from a uniform beurnoulli distribution
    init_methods = [ 'zeros', 'uniform']
    # # different burn-in: turncated version 
    burn_ins = [ 100000, dim0*dim1*1, dim0*dim1*5, dim0*dim1*10] #dim0*dim1*15, dim0*dim1*20 ]
    
    for method in init_methods:
        for burn_len in burn_ins:
            graph = potts_model( dim0, dim1)
            graph.infer_alg = Gibbs(graph, 
                                    init_method=method, 
                                    burn_in=burn_len)
            graph.infer_background( image)







    