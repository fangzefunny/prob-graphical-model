import numpy as np
import pandas as pd
from collections import namedtuple
#name_node = namedtuple( 'node', ('name', 'content'))

'''
This document implement the multiplication of probaility multiplication.
'''

class factor:
    '''
    line 18-line 45 are adopted and modified based on: 
    https://github.com/krashkov/Belief-Propagation/blob/master/2-ImplementationFactor.ipynb

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
        try: 
            self.to_parconfig_dist()
        except: 
            pass 
        
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

    def to_parconfig_dist( self):
        
        new_dist = self.get_distribution()
        new_vars = self.get_variables()
        new_shape = new_dist.shape

        flat_lst = np.reshape( new_dist, [int(new_shape[0]), -1], order='F')
        show_dict = dict() 
        #print( new_vars)
        if new_vars.shape[0]>1:
            other_var = new_vars[1:]
            other_val = np.zeros_like(new_shape[1:])
            other_shape = new_shape[1:]
            rows = 1.
            for dim in other_shape:
                rows *= dim

            for row in range( int(rows)):
                res = row
                par_config = ''
                idx = -1 
                for par_var, dim in zip( other_var, other_shape):
                    par_val = res % dim
                    idx += 1
                    res = int( (res - par_val) / dim)
                    par_config += '{}={},'.format( par_var, par_val + 1)
                show_dict[par_config] = flat_lst[ :, row]
        else:
            show_dict['prior'] = np.reshape(flat_lst, [-1])
        self.parconfig_dist = show_dict
        
    def get_parconfig_dist(self):
        return self.parconfig_dist

    def show_dist( self):
        col = []
        target = self.__variables[0]
        ndim = self.__shape[0]
        for i in range( 1, ndim+1):
            fname = '{}={}'.format( target, i)
            col.append(fname)
        print(pd.DataFrame.from_dict(self.parconfig_dist, 
                                     orient='index', 
                                     columns = col) )
    
    def change_dist(self, target):
        dist = self.get_distribution()
        vars_lst = self.get_variables() 
        #dims  = self.get_shape() 

        target_idx = np.where( vars_lst == target)
        if list(target_idx[0]) != [0]:
            no_target_idx = np.where( vars_lst != target)

            target_mask = np.isin( vars_lst, target, invert=True)

            new_ind = np.array([-1]*len( vars_lst), dtype=int)
            new_ind[ target_mask] = np.arange(np.sum( target_mask))\
                                   + np.sum(np.invert(target_mask))
            new_ind[ target_idx] = np.arange(len(target_idx)) 

            new_vars = list(vars_lst[ target_idx]) + list(vars_lst[ no_target_idx])
            new_dist = np.moveaxis( dist, range(len(new_ind)), new_ind)
        else:
            new_vars = vars_lst
            new_dist = dist

        new_shape = new_dist.shape

        flat_lst = np.reshape( new_dist, [int(new_shape[0]), -1], order='F')
        show_dict = dict() 
        other_var = new_vars[1:]
        other_val = np.zeros_like(new_shape[1:])
        row = 0 
        for n in range(len(new_shape[1:])):
            while other_val[n] < new_shape[n]:
                other_val[n] += 1
                par_config = ''
                for var, val in zip( other_var, other_val):
                    par_config += '{}={},'.format( var, val)
                par_config = par_config[:-1]
                show_dict[par_config] = flat_lst[ :, row]
                row += 1
        print(pd.DataFrame.from_dict(show_dict, 
                                     orient='index', 
                                     columns = range(1, 1+new_shape[0])) )
        
        
def fac_prod(x, y):
    '''
    adopted and modified based on: 
    https://github.com/krashkov/Belief-Propagation/blob/master/2-ImplementationFactor.ipynb

    Implement of product between probability distribution.
    Can automatically do the variables matching.

    The idea of this is to repeat the matrix and move axis to ensure
    that the x and y are with the same dimensions, so we can do
    elementwise product. 

    I tried to create my own function, but it is much slower than this one.
    This is the reasonable why I kept the function in my project. 

    Input:
        x: one distribution
        y: another distribution

    Output:
        x*y in terms of probability
    '''
    if x.is_none() or y.is_none():
        raise Exception('One of the factors is None')
    
    # find the shared, and not shared, variables
    xy, xy_in_x_ind, xy_in_y_ind = np.intersect1d( x.get_variables(), y.get_variables(), return_indices=True)
    
    if xy.size == 0:
        # what is new from the original code
        # created by myself, to improve the original functions to allow multiplication
        # of the probability distribution without sharin variables 
        val, dim = y.get_variables()[0], y.get_shape()[0]
        dist = np.expand_dims( x.get_distribution(), axis = -1)
        new_dist = np.tile( dist, [1, dim])
        new_x = factor( list(x.get_variables())+[val], new_dist)
        return fac_prod( new_x , y)
        
    if not np.all(x.get_shape()[xy_in_x_ind] == y.get_shape()[xy_in_y_ind]):
        raise Exception('Common variables have different order')
    
    # choose the different variables in both distributions, 
    # e.g.  x= p(A|B) y= p(B|C)
    # x_not_in_y: A
    # y_not_in_x: C
    x_not_in_y = np.setdiff1d(x.get_variables(), y.get_variables(), assume_unique=True)
    y_not_in_x = np.setdiff1d(y.get_variables(), x.get_variables(), assume_unique=True)
    
    # creat the placeholder for shared element
    x_mask = np.isin(x.get_variables(), xy, invert=True)
    y_mask = np.isin(y.get_variables(), xy, invert=True)
    
    # new index placeholder for all elements
    x_ind = np.array([-1]*len(x.get_variables()), dtype=int)
    y_ind = np.array([-1]*len(y.get_variables()), dtype=int)
    
    # put the shared element to the end in x, to the head in y (but for indices)
    x_ind[x_mask] = np.arange(np.sum(x_mask))
    y_ind[y_mask] = np.arange(np.sum(y_mask)) + np.sum(np.invert(y_mask))
    
    x_ind[xy_in_x_ind] = np.arange(len(xy)) + np.sum(x_mask)
    y_ind[xy_in_y_ind] = np.arange(len(xy))
    
    # move axis to create the new distribution. 
    x_distribution = np.moveaxis(x.get_distribution(), range(len(x_ind)), x_ind)
    y_distribution = np.moveaxis(y.get_distribution(), range(len(y_ind)), y_ind)

    # multiply between two distributions 
    # e.g p(A|B) * p(B|C)   
    #  A: dim1, B: dim2, C: dim3
    #  dims(p(A|B)) ==> (2,2) --> (2,2,1)
    #  dims(p(B|C)) ==> (2,2) --> (1,2,2)
    res_distribution =   x_distribution[tuple([slice(None)]*len(x.get_variables())+[None]*len(y_not_in_x))] \
                       * y_distribution[tuple([None]*len(x_not_in_y)+[slice(None)])]
    
    return factor(list(x_not_in_y)+list(xy)+list(y_not_in_x), res_distribution)
    
def fac_div(x, y):
    '''
    Implement of product between probability distribution.
    Can automatically do the variables matching.

    The same as the fac_prod, but a divid version. 
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
    '''
    adopted and modified based on: 
    https://github.com/krashkov/Belief-Propagation/blob/master/2-ImplementationFactor.ipynb

    Implement of mariginal over probability distribution.
    sum_x p(x,y) = fac_sum( p_XY, [x])  
    '''
    variables = np.array( variables)
    
    if not np.all(np.in1d(variables, x.get_variables())):
        raise Exception('Factor do not contain given variables')
    
    # Find the other variables 
    res_variables    = np.setdiff1d(x.get_variables(), variables, assume_unique=True)
    res_distribution = np.sum(x.get_distribution(),
                              tuple(np.where(np.isin(x.get_variables(), variables))[0]))
    
    return factor(res_variables, res_distribution)

def fac_max(x, variables):
    '''
    Implement of maximization over probability distribution.

    Implement of max over probability distribution. used in max-produc
    inference 
    sum_x p(x,y) = fac_max( p_XY, [x])  
    '''
    variables = np.array( variables)
    
    if not np.all(np.in1d(variables, x.get_variables())):
        raise Exception('Factor do not contain given variables')

    res_variables    = np.setdiff1d(x.get_variables(), variables, assume_unique=True)
    res_distribution = np.max(x.get_distribution(),
                              tuple(np.where(np.isin(x.get_variables(), variables))[0]))

    return factor(res_variables, res_distribution)

def fac_take(x, variable, value):
    '''
    Adopted and modified based on: 
    https://github.com/krashkov/Belief-Propagation/blob/master/2-ImplementationFactor.ipynb

    Choose the value of the probability.

    Implement of mariginal over probability distribution.
    sum_x p(x=1,y) = fac_sum( p_XY, x, 1)  
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

def prob_matrix( xdict, dims):
    '''
    Created by myself
    '''
    x_lst = []
    for key in xdict.keys():
        x_lst += xdict[ key]
    x_lst = np.array(x_lst)
    return np.reshape( x_lst, dims, order='F')

def normalize( dist):
    '''
    Normalization, created by myself
    '''
    return dist/np.sum(dist)

def to_bin(x, ndim):
    a = bin(x)
    num0 = ndim + 2 - len(a)
    return a.replace('0b', '' + '0' * num0)

def uni_sample( prob):
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

def sigmoid( x):
    return 1/(1 + np.exp(-x))
        