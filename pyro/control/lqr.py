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
from pyro.control  import controller
from pyro.dynamic  import statespace
from pyro.analysis import costfunction
##############################################################################


#################################################################
def synthesize_lqr_controller( ss , cf , xbar = None, ubar = None):
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
    xbar: offset on state feedback
    ubar: offset on control input
        
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
    
    # Define linear controller
    ctl = linear.ProportionalController.from_matrix( K )
    ctl.name = 'LQR controller'
    
    # Offsets on input and ouputs
    if xbar is not None:
        ctl.ybar = xbar  # Offset on the sensor signal
        
    if ubar is not None:
        ctl.ubar = ubar  # Offset on the control input
    
    
    return ctl


#################################################################
def linearize_and_synthesize_lqr_controller( sys , cf ):
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
    sys : `ContinuousSystem` instance
    cf  : "quadratic cost function" instance
        
    Returns
    -------
    instance of `Proportionnal Controller`

    """
    
    ss  = statespace.linearize( sys , 0.01 )
    
    ctl = synthesize_lqr_controller( ss , cf , sys.xbar , sys.ubar )
    
    return ctl



##############################################################################
### Time varying LQR controller
##############################################################################

class TrajectoryLQRController( controller.StaticController ):
    """
    General (SISO or MIMO) proportional controller
    -------------------------------------------------

    u = u_d(t) + K(t) ( xd(t) - x )
    
    
    K = R^(-1) B(t) S(t)
    
    S(t) = integral of differential riccati equation
    
    A(t) = df/dx(t)
    B(t) = df/du(t)
    
    x = y  : state feedback is assumed
    
    -----------------------------------------
    r  : reference signal vector       k x 1
    y  : sensor signal vector          p x 1
    u  : control inputs vector         m x 1
    -----------------------------------------
    
    """
    
    ###############################
    def __init__(self, sys , traj , cf = None ):
        
        n = sys.n # states dimensions
        m = sys.m # input dimenstions
        
        controller.StaticController.__init__( self, n, m, n)
        
        self.name = "Trajectory LQR Controller"
        
        self.sys  = sys
        self.traj = traj
        
        if cf is None:
            
            self.cf = sys.cost_function
            
        else:
            
            self.cf = cf
            
        self.compute_linear_dynamic()
        self.compute_cost_matrices()
        self.compute_cost2go()
        self.compute_gains()
        
        self.traj.generate_interpol_functions()
        
        print('\n\n-------------------------------------------------------------------------')
        print('Warning: this method is an approximation of the riccati solution for now')
        print('ToDo @Alex : fix differential riccati integration')
        print('-------------------------------------------------------------------------\n')
            
            
    ###############################
    def compute_linear_dynamic(self):
        
        steps = self.traj.time_steps
        
        self.As = np.zeros(( self.sys.n , self.sys.n , steps ))
        self.Bs = np.zeros(( self.sys.n , self.sys.m , steps ))
        
        for i in range(steps):
            
            self.sys.tbar = self.traj.t[i]
            self.sys.xbar = self.traj.x[i,:]
            self.sys.ubar = self.traj.u[i,:]
            
            # TODO more intelligent gradient computation
            ss  = statespace.linearize( self.sys , 0.01 )
            
            self.As[:,:,i] = ss.A
            self.Bs[:,:,i] = ss.B
            
            
    
    ###############################
    def compute_cost_matrices(self):
        
        steps = self.traj.time_steps
        
        self.Qs = np.zeros(( self.sys.n , self.sys.n , steps ))
        self.Rs = np.zeros(( self.sys.m , self.sys.m , steps ))
        
        for i in range(steps):
            
            # ToDo Add Time Varying Quadratic cost object
            self.Qs[:,:,i] = self.cf.Q
            self.Rs[:,:,i] = self.cf.R
        
        
        
    ###############################
    def compute_cost2go(self):
        
        steps = self.traj.time_steps
        
        self.Ss = np.zeros(( self.sys.n , self.sys.n , steps ))
        self.Ps = np.zeros(( self.sys.n , self.sys.n , steps ))
        
        # Boundary conditions
        Af = self.As[:,:,-1]
        Bf = self.Bs[:,:,-1]
        Qf = self.Qs[:,:,-1]
        Rf = self.Rs[:,:,-1]
        
        self.Ss[:,:,-1] = solve_continuous_are( Af , Bf , Qf , Rf )
        
        # self.Ss[:,:,-1] = np.eye( self.sys.n )
        # self.Ps[:,:,-1] = np.eye( self.sys.n )
        
        for i in range( steps -1 , 0, -1 ):
            
            A = self.As[:,:,i]
            B = self.Bs[:,:,i]
            Q = self.Qs[:,:,i]
            R = self.Rs[:,:,i]
            S = self.Ss[:,:,i]
            # P = self.Ps[:,:,i]
            
            # Quick approx for testing
            dS = Q 
            
            ##############
            ## TODO: Make a real integraiton of differential riccati equation
            #############
            
            # dS = S @ A + A.T @ S - S @ B @ np.linalg.inv(R) @ B.T @ S + Q
            # dP = A.T @ P - 0.5 * S @ B @ np.linalg.inv(R) @ B.T @ P + 0.5 * Q @ np.linalg.inv(P).T
            
            dt = self.traj.t[i] - self.traj.t[i-1]
            
            # Euler integration (maybe is not engouh precision)
            self.Ss[:,:,i-1] = self.Ss[:,:,i] + dS * dt
            
            # self.Ps[:,:,i-1] = self.Ps[:,:,i] + dP * dt
            # self.Ss[:,:,i-1] = self.Ps[:,:,i-1] @ self.Ps[:,:,i-1].T
            
            # print('t = ', self.traj.t[i])
            # print('\ndS = \n' , dS)
            # print('\nS = \n' , S)
            
            
    ###############################
    def compute_gains(self):
        
        steps = self.traj.time_steps
        
        self.Ks = np.zeros(( self.sys.m , self.sys.n , steps ))
        
        for i in range( steps ):
            
            A = self.As[:,:,i]
            B = self.Bs[:,:,i]
            Q = self.Qs[:,:,i]
            R = self.Rs[:,:,i]
            S = self.Ss[:,:,i]
            
            K = np.linalg.inv(R) @ B.T @  S
            
            print('\n-------------------\nt = ', self.traj.t[i])
            print('K = \n' , K)
            
            self.Ks[:,:,i] = K
        
        
    ##############################
    def c(self, x, r, t):
        """ Feedback law """
        
        if t < self.traj.time_final:
            
            # Nominal control input
            u_bar = self.traj.inter_t2u( t )
            x_bar = self.traj.inter_t2x( t )
            
            # Find closest traj point
            i = (np.abs(self.traj.t - t)).argmin()
            
            K = self.Ks[:,:,i]  # Time varying linear gain
            
            u_e = K @ ( x_bar - x )
            
            u = u_bar + u_e
            
        else:
            
            x_bar = self.traj.x[-1,:]
            
            K = self.Ks[:,:,-1]
            
            u = K @ ( x_bar - x ) # this assume final position is a fixed point
        
        return u


    


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":
    
    
    from pyro.dynamic.pendulum      import DoublePendulum
    from pyro.analysis.costfunction import QuadraticCostFunction
    
    sys = DoublePendulum()
    
    ss  = statespace.linearize( sys , 0.01 )
    
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
    