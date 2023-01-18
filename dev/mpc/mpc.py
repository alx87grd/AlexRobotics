#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 03:14:40 2021

@author: alex
"""

import numpy as np

from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation
from pyro.dynamic.statespace              import linearize


from pyro.control import controller

##########################################################################
#T
###########################################################################
class NonLinearMPC( controller.StaticController ) :
    """ 

    """
    ############################
    def __init__(self, sys, dt = 0.1 , n = 10 ):
        """ """
        
        # Dimensions
        self.k = 1   
        self.m = sys.m   
        self.p = sys.p
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'MPC Controller'
        
        # Gains
        self.goal    = sys.xbar
        
        self.planner = DirectCollocationTrajectoryOptimisation( sys , dt , n )
        
        self.planner.x_goal = self.goal
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor vector     p x 1
        r  : reference vector  k x 1
        t  : time              1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        self.planner.x_start = y  # actual state
        
        self.planner.compute_optimal_trajectory()
        
        x,u = self.planner.decisionvariables2xu( self.planner.res.x )
        
        #DEBUG
        self.planner.show_solution()
        print(self.planner.x_start)
        
        u_next = u[:,0]
        
        return u_next
    
    
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    from pyro.dynamic.massspringdamper        import TwoMass
    from pyro.analysis.costfunction import QuadraticCostFunction
    
    sys = TwoMass()
    
    #Full state feedback (default of class is x2 output only)
    sys.C = np.diag([1,1,1,1])
    sys.p = 4 # dim of output vector
    sys.output_label = sys.state_label
    sys.output_units = sys.state_units
    
    # Cost function
    cf  = QuadraticCostFunction.from_sys( sys )
    cf.Q[0,0] = 1
    cf.Q[1,1] = 1
    cf.Q[2,2] = 1
    cf.Q[3,3] = 1
    cf.R[0,0] = 1
    sys.cost_function = cf
    
    sys.u_ub[0] = +2
    sys.u_lb[0] = 0
    
    ctl = NonLinearMPC( sys , 0.1 , 30)
    
    ctl.planner.goal    = np.array([0,0,0,0])
    ctl.planner.maxiter = 10
    
    # New cl-dynamic
    cl_sys = ctl + sys
    
    cl_sys.x0 = np.array([0.5,0.5,0,0])
    cl_sys.compute_trajectory(0.3,4,'euler')
    cl_sys.plot_trajectory('xu')
    cl_sys.animate_simulation()