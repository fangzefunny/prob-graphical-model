import numpy as np 
import pandas as pd
from utils import factor, fac_prod, fac_sum, fac_max, fac_div, fac_take, prob_matrix

# p(x1)
p_x1 = { 'prior': [.4, .6]} 
p_X1 = factor( ['X1'], prob_matrix(p_x1, [2,]))
p_X1.show_dist()

# p(x2|x1)
p_x2x1 = { 'X1=0': [.3, .7],
         'X1=1': [.8, .2]} 
p_X2X1 = factor( ['X2','X1'], prob_matrix(p_x2x1, [2,2]))
p_X2X1.show_dist()

# p(x3|x1)
p_x3x1 = { 'X1=0': [.8, .2],
           'X1=1': [.3, .7]} 
p_X3X1 = factor( ['X3','X1'], prob_matrix(p_x3x1, [2,2]))
p_X3X1.show_dist()

# p(D|x1)
p_dx1 = { 'X1=0': [.7, .3],
          'X1=1': [.4, .6]} 
p_DX1 = factor( ['D','X1'], prob_matrix(p_dx1, [2,2]))
p_DX1.show_dist()

# U
u = { 'X3=0,D=0': [50],
      'X3=1,D=0': [70],
      'X3=0,D=1': [40],
      'X3=1,D=1': [100]}
U = factor( ['u','X3','D'], prob_matrix(u, [1,2,2]))
U.show_dist()

# remove X3
U = fac_sum( fac_prod(p_X3X1, U), 'X3')
U.show_dist()

# remove d
U0, d_star1 = fac_max( fac_take( U, 'X1', 0), ['D'])
U0.show_dist()
U1, d_star2 = fac_max( fac_take( U, 'X1', 1), ['D'])
U1.show_dist()
print( 'optimal policy:', [d_star1, d_star2])

print( 'MEU:', U0.get_distribution() + U1.get_distribution())

U2,d_star=fac_max( U, 'D')
U2.show_dist()
print( 'optimal policy:', [d_star])



# # joint
# joint = fac_prod( fac_prod( p_X1, p_X2X1), p_X3X1)
# joint.show_dist()

# #p(d,e)
# pde = fac_sum( joint, ['X1', 'X3'])
# pde.show_dist()

# #f1
# f1 = fac_div( joint, pde)
# f1 = fac_take( f1, 'X2', 1)
# print( 'p(x1, x3|x2=1) :', f1.get_distribution())
p1 = .4
p2 = .6
p3 = .7
p4 = .8
transition = { 'high, search':       [ p1, 1 - p1, 0],
            'low, search':        [ 0, p2, 1 - p2],
            'exhausted, search':  [ 0, 0, 1],
            'high, wait':         [ 1, 0, 0], 
            'low, wait':          [ p3, 1-p3, 0],
            'exhausted, wait':    [ 0, p4, 1 - p4]}
for key in transition:
    print(key)