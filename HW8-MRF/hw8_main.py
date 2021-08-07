import numpy as np 
import pandas as pd
from utils import factor, fac_prod, fac_sum, fac_max, fac_div, fac_take, prob_matrix, normalize 

#max_product

#init message 
# msg_ac = factor( ['A'], np.array([ 1., 1.]))
# msg_bc = factor( ['B'], np.array([ 0., 1.]))
# msg_be = factor( ['B'], np.array([ 0., 1.]))
# msg_ca = factor( ['C'], np.array([ 1., 1.]))
# msg_cb = factor( ['C'], np.array([ 1., 1.]))
# msg_cd = factor( ['C'], np.array([ 1., 1.]))
# msg_dc = factor( ['D'], np.array([ 1., 1.]))
# msg_eb = factor( ['E'], np.array([ 1., 1.]))

# # init unary function 
# phi_A = factor( ['A'], np.exp([ -2., 1.2]))
# phi_B = factor( ['B'], np.array([ 0., 1.]))
# phi_C = factor( ['C'], np.exp([ -.3, 2.]))
# phi_D = factor( ['D'], np.exp([ 1.3, -.8]))
# phi_E = factor( ['E'], np.exp([ .2, -.2]))

# # init pairwise 
# psi_ac = { 'C=-1': [np.exp(2), np.exp(-1)],
#            'C= 1': [np.exp(-1), np.exp(2)]}
# psi_ac  = factor( ['A', 'C'], prob_matrix(psi_ac, [2,2]))

# psi_bc = { 'C=-1': [np.exp(-.3), np.exp(1.2)],
#            'C= 1': [np.exp(1.2), np.exp(-.3)]}
# psi_bc  = factor( ['B', 'C'], prob_matrix(psi_bc, [2,2]))

# psi_cd = { 'C=-1': [np.exp(.5), np.exp(-1.2)],
#            'C= 1': [np.exp(-1.2), np.exp(.5)]}
# psi_cd  = factor( ['D', 'C'], prob_matrix(psi_cd, [2,2]))

# psi_be = { 'E=-1': [np.exp(-1), np.exp(.7)],
#            'E= 1': [np.exp(.7), np.exp(-1)]}
# psi_be  = factor( ['B', 'E'], prob_matrix(psi_be, [2,2]))

# bel_a = np.array([.5, .5])
# bel_c = np.array([.5, .5])
# bel_d = np.array([.5, .5])
# bel_e = np.array([.5, .5])

# for i_iter in range(3):

#     old_bels = np.array( [bel_a, bel_c, bel_d, bel_e])

#     print( '=======> iteration {} =========>'.format( i_iter))

#     #for node A:
#     # calculate message 
#     msg_ac = fac_sum( fac_prod( fac_prod( phi_C, psi_ac), fac_prod( msg_cb, msg_cd)), ['C'])
#     print( 'message ac:', msg_ac.get_variables(), msg_ac.get_distribution())
#     # calculate belief
#     bel_a = normalize( fac_prod( phi_A, msg_ac).get_distribution())
#     print( 'belief A:', bel_a)

#     #for node C:
#     # calculate message ca 
#     msg_ca = fac_sum( fac_prod( phi_A, psi_ac), ['A'])
#     print( 'message ca:', msg_ca.get_variables(), msg_ca.get_distribution())
#     msg_cb = fac_sum( fac_prod( fac_prod( phi_B, psi_bc), msg_be), ['B'])
#     print( 'message cb:', msg_cb.get_variables(), msg_cb.get_distribution())
#     msg_cd = fac_sum( fac_prod( phi_D, psi_cd), ['D'])
#     print( 'message cd:', msg_cd.get_variables(), msg_cd.get_distribution())
#     # calculate belief
#     bel_c = normalize( fac_prod( phi_C, fac_prod(fac_prod( msg_ca, msg_cb), msg_cd)).get_distribution())
#     print( 'belief C:', bel_c)

#     #for node D:
#     # calculate message ca 
#     msg_dc = fac_sum( fac_prod( fac_prod( phi_C, psi_cd), fac_prod( msg_ca, msg_cb)), ['C'])
#     print( 'message dc:', msg_dc.get_variables(), msg_dc.get_distribution())
#     # calculate belief
#     bel_d = normalize( fac_prod( phi_D, msg_dc).get_distribution())
#     print( 'belief D:', bel_d)

#     #for node E:
#     # calculate message ca 
#     msg_eb = fac_sum( fac_prod( fac_prod( phi_B, psi_be), msg_bc), ['B'])
#     print( 'message eb:', msg_eb.get_variables(), msg_eb.get_distribution())
#     # calculate belief
#     bel_e = normalize( fac_prod( phi_E, msg_eb).get_distribution())
#     print( 'belief E:', bel_e)

#     delta = np.sum( abs(np.array( [ bel_a, bel_c, bel_d, bel_e]) - old_bels))
#     print('convergence:', delta)

# for i_iter in range(3):

#     old_bels = np.array( [bel_a, bel_c, bel_d, bel_e])

#     print( '=======> iteration {} =========>'.format( i_iter))

#     #for node A:
#     # calculate message 
#     msg_ac = fac_max( fac_prod( fac_prod( phi_C, psi_ac), fac_prod( msg_cb, msg_cd)), ['C'])
#     print( 'message ac:', msg_ac.get_variables(), msg_ac.get_distribution())
#     # calculate belief
#     bel_a = normalize( fac_prod( phi_A, msg_ac).get_distribution())
#     print( 'belief A:', bel_a)

#     #for node C:
#     # calculate message ca 
#     msg_ca = fac_max( fac_prod( phi_A, psi_ac), ['A'])
#     print( 'message ca:', msg_ca.get_variables(), msg_ca.get_distribution())
#     msg_cb = fac_max( fac_prod( fac_prod( phi_B, psi_bc), msg_be), ['B'])
#     print( 'message cb:', msg_cb.get_variables(), msg_cb.get_distribution())
#     msg_cd = fac_max( fac_prod( phi_D, psi_cd), ['D'])
#     print( 'message cd:', msg_cd.get_variables(), msg_cd.get_distribution())
#     # calculate belief
#     bel_c = normalize( fac_prod( phi_C, fac_prod(fac_prod( msg_ca, msg_cb), msg_cd)).get_distribution())
#     print( 'belief C:', bel_c)

#     #for node D:
#     # calculate message ca 
#     msg_dc = fac_max( fac_prod( fac_prod( phi_C, psi_cd), fac_prod( msg_ca, msg_cb)), ['C'])
#     print( 'message dc:', msg_dc.get_variables(), msg_dc.get_distribution())
#     # calculate belief
#     bel_d = normalize( fac_prod( phi_D, msg_dc).get_distribution())
#     print( 'belief D:', bel_d)

#     #for node E:
#     # calculate message ca 
#     msg_eb = fac_max( fac_prod( fac_prod( phi_B, psi_be), msg_bc), ['B'])
#     print( 'message eb:', msg_eb.get_variables(), msg_eb.get_distribution())
#     # calculate belief
#     bel_e = normalize( fac_prod( phi_E, msg_eb).get_distribution())
#     print( 'belief E:', bel_e)

#     delta = np.sum( abs(np.array( [ bel_a, bel_c, bel_d, bel_e]) - old_bels))
#     print('convergence:', delta)

# msg_ac = factor( ['A'], np.array([ 1., 1.]))
# msg_bc = factor( ['B'], np.array([ 1., 1.]))
# msg_ca = factor( ['C'], np.array([ 0., 1.]))
# msg_cb = factor( ['C'], np.array([ 0., 1.]))
# msg_cd = factor( ['C'], np.array([ 0., 1.]))
# msg_dc = factor( ['D'], np.array([ 1., 1.]))

# # init unary function 
# phi_A = factor( ['A'], np.exp([ -1.2, 2.]))
# phi_B = factor( ['B'], np.exp([ .8, -.2]))
# phi_C = factor( ['C'], np.array([0., 1]))
# phi_D = factor( ['D'], np.exp([ 0.2, -.2]))

# # init pairwise 
# psi_ac = { 'C=-1': [np.exp(2), np.exp(-1)],
#            'C= 1': [np.exp(-1), np.exp(2)]}
# psi_ac  = factor( ['A', 'C'], prob_matrix(psi_ac, [2,2]))

# psi_bc = { 'C=-1': [np.exp(-.3), np.exp(1.2)],
#            'C= 1': [np.exp(1.2), np.exp(-.3)]}
# psi_bc  = factor( ['B', 'C'], prob_matrix(psi_bc, [2,2]))

# psi_cd = { 'C=-1': [np.exp(.9), np.exp(-.2)],
#            'C= 1': [np.exp(-.2), np.exp(.9)]}
# psi_cd  = factor( ['D', 'C'], prob_matrix(psi_cd, [2,2]))


# bel_a = np.array([.5, .5])
# bel_b = np.array([.5, .5])
# bel_d = np.array([.5, .5])

# for i_iter in range(3):

#     old_bels = np.array( [bel_a, bel_b, bel_d])

#     print( '=======> iteration {} =========>'.format( i_iter))

#     #for node A:
#     # calculate message 
#     msg_ac = fac_sum( fac_prod( fac_prod( phi_C, psi_ac), fac_prod( msg_cb, msg_cd)), ['C'])
#     print( 'message ac:', msg_ac.get_variables(), msg_ac.get_distribution())
#     # calculate belief
#     bel_a = normalize( fac_prod( phi_A, msg_ac).get_distribution())
#     print( 'belief A:', bel_a)

#     #for node B:
#     # calculate message ca 
#     msg_bc = fac_sum( fac_prod( fac_prod( phi_C, psi_bc), fac_prod( msg_ca, msg_cd)), ['C'])
#     print( 'message bc:', msg_bc.get_variables(), msg_bc.get_distribution())
#     # calculate belief
#     bel_b = normalize( fac_prod( phi_B, msg_bc).get_distribution())
#     print( 'belief B:', bel_b)

#     #for node D:
#     # calculate message ca 
#     msg_dc = fac_sum( fac_prod( fac_prod( phi_C, psi_cd), fac_prod( msg_ca, msg_cb)), ['C'])
#     print( 'message dc:', msg_dc.get_variables(), msg_dc.get_distribution())
#     # calculate belief
#     bel_d = normalize( fac_prod( phi_D, msg_dc).get_distribution())
#     print( 'belief D:', bel_d)

#     delta = np.sum( abs(np.array( [ bel_a, bel_b, bel_d]) - old_bels))
#     print('convergence:', delta)


###### Fg
# init unary function 
F_a = factor( ['A'], np.exp([ -2., 1.2]))
F_b = factor( ['B'], np.array([ 0., 1.]))
F_c = factor( ['C'], np.exp([ -.3, 2.]))
F_d = factor( ['D'], np.exp([ 1.3, -.8]))
F_e = factor( ['E'], np.exp([ .2, -.2]))
F_ac = { 'C=-1': [np.exp(2), np.exp(-1)],
         'C= 1': [np.exp(-1), np.exp(2)]}
F_ac  = factor( ['A', 'C'], prob_matrix(F_ac, [2,2]))

F_bc = { 'C=-1': [0, np.exp(1.2)],
         'C= 1': [0, np.exp(-.3)]}
F_bc  = factor( ['B', 'C'], prob_matrix(F_bc, [2,2]))

F_cd = { 'C=-1': [np.exp(.5), np.exp(-1.2)],
         'C= 1': [np.exp(-1.2), np.exp(.5)]}
F_cd  = factor( ['D', 'C'], prob_matrix(F_cd, [2,2]))

F_be = { 'E=-1': [0, np.exp(.7)],
         'E= 1': [0, np.exp(-1)]}
F_be  = factor( ['B', 'E'], prob_matrix(F_be, [2,2]))

msg_xa_Fa  = factor( ['A'], np.array([ 1, 1]))
msg_xb_Fb  = factor( ['B'], np.array([ 1, 1]))
msg_xc_Fc  = factor( ['C'], np.array([ 1, 1]))
msg_xd_Fd  = factor( ['D'], np.array([ 1, 1]))
msg_xe_Fe  = factor( ['E'], np.array([ 1, 1]))
msg_xa_Fac = factor( ['A'], np.array([ 1, 1]))
msg_xb_Fbe = factor( ['B'], np.array([ 1, 1]))
msg_xb_Fbc = factor( ['B'], np.array([ 1, 1]))
msg_xc_Fac = factor( ['C'], np.array([ 1, 1]))
msg_xc_Fbc = factor( ['C'], np.array([ 1, 1]))
msg_xc_Fcd = factor( ['C'], np.array([ 1, 1]))
msg_xd_Fcd = factor( ['D'], np.array([ 1, 1]))
msg_xe_Fbe = factor( ['E'], np.array([ 1, 1]))

msg_Fa_xa  = F_a
msg_Fb_xb  = F_b
msg_Fc_xc  = F_c
msg_Fd_xd  = F_d
msg_Fe_xe  = F_e
msg_Fac_xa = factor( ['A'], np.array([ 1, 1]))
msg_Fbc_xb = factor( ['B'], np.array([ 1, 1]))
msg_Fbe_xb = factor( ['B'], np.array([ 1, 1]))
msg_Fac_xc = factor( ['C'], np.array([ 1, 1]))
msg_Fbc_xc = factor( ['C'], np.array([ 1, 1]))
msg_Fcd_xc = factor( ['C'], np.array([ 1, 1]))
msg_Fcd_xd = factor( ['D'], np.array([ 1, 1]))
msg_Fbe_xe = factor( ['E'], np.array([ 1, 1]))

bel_a = np.array([.5, .5])
bel_c = np.array([.5, .5])
bel_d = np.array([.5, .5])
bel_e = np.array([.5, .5])
i_iter = 0
done = False

while not done:

    old_bels = np.array( [bel_a, bel_c, bel_d, bel_e])

    print( '========= iteration: {} =========='.format(i_iter))



    # cal  msg: vars --> function:
  
    # for a 
    msg_xa_Fa = msg_Fac_xa
    print('msg_Fxa_Fa', msg_xa_Fa.get_distribution())
    msg_xa_Fac = msg_Fa_xa
    print('msg_xa_Fac', msg_xa_Fac.get_distribution())
    # for b 
    msg_xb_Fb = fac_prod( msg_Fbc_xb, msg_Fbe_xb)
    print('msg_xb_Fb', msg_xb_Fb.get_distribution())
    msg_xb_Fbc = fac_prod( msg_Fb_xb, msg_Fbe_xb)
    print('msg_xb_Fbc', msg_xb_Fbc.get_distribution())
    msg_xb_Fbe = fac_prod( msg_Fb_xb, msg_Fbe_xb)
    print('msg_xb_Fbe', msg_xb_Fbe.get_distribution())
    # for c
    msg_xc_Fc = fac_prod( fac_prod(msg_Fac_xc, msg_Fbc_xc), msg_Fcd_xc)
    print('msg_xc_Fc', msg_xc_Fc.get_distribution())
    msg_xc_Fac = fac_prod( fac_prod(msg_Fc_xc, msg_Fbc_xc), msg_Fcd_xc)
    print('msg_xc_Fac', msg_xc_Fac.get_distribution())
    msg_xc_Fbc = fac_prod( fac_prod(msg_Fc_xc, msg_Fac_xc), msg_Fcd_xc)
    print('msg_xc_Fbc', msg_xc_Fbc.get_distribution())
    msg_xc_Fcd = fac_prod( fac_prod(msg_Fc_xc, msg_Fac_xc), msg_Fbc_xc)
    print('msg_xc_Fcd', msg_xc_Fcd.get_distribution())
    # for d
    msg_xd_Fd = msg_Fcd_xd
    print('msg_xd_Fd', msg_xd_Fd.get_distribution())
    msg_xd_Fcd = msg_Fd_xd
    print('msg_xd_Fcd', msg_xd_Fcd.get_distribution())
    # for e
    msg_xe_Fe = msg_Fbe_xe
    print('msg_xe_Fe', msg_xe_Fe.get_distribution())
    msg_xe_Fbe = msg_Fe_xe
    print('msg_xe_Fbe', msg_xe_Fbe.get_distribution())
        
        
    
    # init msg: vars --> function 
    # for A 
    msg_Fa_xa  = F_a
    print('msg_Fa_xa',msg_Fa_xa.get_distribution())
    msg_Fac_xa = fac_max( fac_prod(F_ac, msg_xc_Fac), ['C'])
    print('msg_Fac_xa',msg_Fac_xa.get_distribution())
    bel_a = normalize(fac_prod(msg_Fa_xa, msg_Fac_xa).get_distribution())
    print( 'bel_a', bel_a)
    # for b
    msg_Fb_xb = F_b
    print( 'msg_Fb_xb:', msg_Fb_xb.get_distribution())
    # for c 
    msg_Fc_xc = F_c
    print( 'msg_Fc_xc:', msg_Fc_xc.get_distribution())
    msg_Fac_xc = fac_max( fac_prod(F_ac, msg_xa_Fac), ['A'])
    print('msg_Fac_xc',msg_Fac_xc.get_distribution())
    msg_Fbc_xc = fac_max( fac_prod(F_bc, msg_xb_Fbc), ['B'])
    print('msg_Fbc_xc',msg_Fbc_xc.get_distribution())
    msg_Fcd_xc = fac_max( fac_prod(F_cd, msg_xd_Fcd), ['D'])
    print('msg_Fcd_xc',msg_Fcd_xc.get_distribution())
    bel_c = normalize(fac_prod(fac_prod(msg_Fc_xc, msg_Fac_xc), fac_prod(msg_Fbc_xc, msg_Fcd_xc)).get_distribution())
    print( 'bel_c', bel_c)
    # for d
    msg_Fd_xd = F_d
    print( 'msg_Fd_xd', msg_Fd_xd.get_distribution())
    msg_Fcd_xd = fac_max( fac_prod(F_cd, msg_xc_Fcd), ['C'])
    print('msg_Fcd_xd',msg_Fcd_xd.get_distribution())
    bel_d = normalize(fac_prod(msg_Fd_xd, msg_Fcd_xd).get_distribution())
    print( 'bel_d', bel_d)
    # for e 
    msg_Fe_xe = F_e
    print( 'msg_Fe_xe', msg_Fe_xe.get_distribution())
    msg_Fbe_xe = fac_max( fac_prod(F_be, msg_xb_Fbe), ['B'])
    print('msg_Fbe_xe',msg_Fbe_xe.get_distribution())
    bel_e = normalize(fac_prod(msg_Fe_xe, msg_Fbe_xe).get_distribution())
    print( 'bel_e', bel_e)

    delta = np.sum( abs(np.array( [ bel_a, bel_c, bel_d, bel_e]) - old_bels))
    print('convergence:', delta)

    i_iter += 1 

    if delta < 1e-3:
        done = True



