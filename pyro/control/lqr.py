#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:05:15 2020

@author: alex
"""

##############################################################################
import numpy as np
from scipy.linalg  import solve_continuous_are

##############################################################################
from pyro.control  import linear
from pyro.dynamic  import statespace
from pyro.analysis import costfunction
##############################################################################


#################################################################
def synthesize_lqr_controller( ss , cf ):
    """

    Compute the optimal linear controller minimizing the quadratic cost:
        
    J = int ( xQx + uRu ) dt = xSx
    
    with control law:
        
    u = K x
    
    Note:
    ---------
    Controller assume y = x  (output is directly the state vector)

    Parameters
    ----------
    sys : `StateSpaceSystem` instance
    cf  : "quadratic cost function" instance
        
    Returns
    -------
    instance of `Proportionnal Controller`

    """
    
    # Solve Riccati Equation
    # A’X + XA - XBR^-1B’X+Q=0
    S = solve_continuous_are( ss.A , ss.B , cf.Q , cf.R )
    
    # Matrix gain
    BTS   = np.dot( ss.B.T , S )
    R_inv = np.linalg.inv( cf.R )
    K     = np.dot( R_inv  , BTS )
    
    ctl = linear.ProportionalController( K )
    ctl.name = 'LQR controller'
    
    return ctl
    


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":
    
    
    from pyro.dynamic.pendulum      import DoublePendulum
    from pyro.analysis.costfunction import QuadraticCostFunction
    from pyro.dynamic.statespace    import linearize
    
    sys = DoublePendulum()
    
    ss  = linearize( sys , 0.01 )
    
    cf  = QuadraticCostFunction.from_sys( sys )
    
    # Tune cost function here:
    cf.R[0,0] = 1000
    cf.R[1,1] = 10000
    
    ctl = synthesize_lqr_controller( ss , cf )
    
    print('\nLinearized sys matrix A:')
    print(ss.A)
    print('\nLinearized sys matrix B:')
    print(ss.B)
    
    print('\nCost function matrix Q:')
    print(cf.Q)
    print('\nCost function matrix R:')
    print(cf.R)
    
    print('\nLQR Controller gain K:')
    print(ctl.K)
    
    
    # Simulation Open-Loop Non-linear
    sys.x0 = np.array([0.1,0.2,0,0])
    sys.plot_trajectory()
    sys.animate_simulation()
    
    # Simulation Open-Loop Linearized
    ss.x0 = np.array([0.1,0.2,0,0])
    ss.compute_trajectory()
    sys.traj = ss.traj
    sys.animate_simulation()
    
    # Simulation Closed-Loop Non-linear with LQR controller
    cl_sys = ctl + sys
    cl_sys.x0 = np.array([0.1,0.2,0,0])
    cl_sys.compute_trajectory()
    cl_sys.plot_trajectory('xu')
    cl_sys.animate_simulation()
    