#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 20:27:58 2023

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

from pyro.planning import dynamicprogramming 

from functionapproximation import LinearFunctionApproximator


###############################################################################
### DP Algo
###############################################################################

class LinearApproximateDynamicProgramming( dynamicprogramming.DynamicProgrammingWithLookUpTable ):
    """  """
    
    ############################
    def __init__(self, grid_sys , cost_function , function_approximation , t = 0):
        
        # Dynamic system
        self.grid_sys  = grid_sys         # Discretized Dynamic system class
        self.sys       = grid_sys.sys     # Base Dynamic system class
        
        # Function approx
        self.fa = function_approximation
        
        # Cost function
        self.cf  = cost_function
        self.t   = t
        
        # Options
        self.alpha                = 0.9    # exponential forgetting factor
        self.gamma                = 0.2
        self.save_time_history    = True
        
        # Memory Variables
        self.k = 0         # Number of computed steps
        
        # Start time (needed to plot elapsed computation time)
        self.start_time = time.time()
        
        # Init params
        self.w =  np.zeros( self.fa.n )  # Initial params
        
        self.compute_cost_lookuptable() 
        
        self.compute_kernels()
        
        self.J = self.P.T @ self.w    # Initial J approx on the grid
        
        # 
        if self.save_time_history:

            self.w_list  = []
            self.J_list  = []
            self.t_list  = []
            
            # Value at t = t_f
            self.J_list.append(  self.J  )
            self.w_list.append(  self.w )
            self.t_list.append(  self.k )
        
        
    ###############################
    def compute_kernels(self):
        """ Compute Kernels """
        
        print('\nComputing Funtion Approximation Kernels:')
        print('-----------------------------------------')
        
        self.P = self.fa.compute_all_kernel( self.grid_sys.state_from_node_id )
        
        m = self.grid_sys.nodes_n * self.grid_sys.actions_n
        n = self.sys.n
        
        Xnext       = self.grid_sys.x_next_table.reshape( ( m , n) )
        self.P_next = self.fa.compute_all_kernel( Xnext )
        
        
        #### Test
        eps   = 0.4 #self.cf.EPS
        xbar  = self.cf.xbar
        
        on_target = np.full( self.grid_sys.x_next_table[:,:,0].shape , True )
        
        for i in range( self.sys.n ):
            
            on_target = np.logical_and( on_target , (self.grid_sys.x_next_table[:,:,i] - xbar[i])**2 < eps )
            
        self.on_target = on_target
        
        #self.off_target = 
        #### 
        
    
    ###############################
    def initialize_backward_step(self):
        """ One step of value iteration """
        
        # Update values
        self.w_old  = self.w
        self.k      = self.k + 1    # index backward in time

                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """
        
        J_next = self.P_next.T @ self.w
        
        Q      = self.G + self.alpha * J_next.reshape( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) )
        
        Q[ self.on_target  ] = 0.            # Test
        Q[ Q > self.cf.INF ] = self.cf.INF   # Test
                        
        J_d    = Q.min( axis = 1 ) # New J Samples
        
        w , J_hat = self.fa.least_square_fit( J_d , self.P )
        
        #e      = self.J - J_d
        #dJ_dw  = self.fa.dJ_dw()
        
        
        
        #self.J = J_hat
        #self.w = w 
        self.w = self.w + self.gamma * ( w - self.w )
        self.J = self.P.T @ self.w
                    
    
    ###############################
    def finalize_backward_step(self):
        """ One step of value iteration """
        
        # Computation time
        elapsed_time = time.time() - self.start_time
        
        # Convergence check 
        j_max     = self.J.max()
        delta     = self.w - self.w_old
        delta_max = delta.max()
        delta_min = delta.min()
        
        print('%d t:%.2f Elasped:%.2f max: %.2f dmax:%.2f dmin:%.2f' % (self.k,self.t,elapsed_time,j_max,delta_max,delta_min) )
        
        # List in memory
        if self.save_time_history:
            self.J_list.append(  self.J  )
            self.w_list.append(  self.w  )
            self.t_list.append(  self.k  )
            
        # return largest J change for usage as stoping criteria
        return abs(np.array([delta_max,delta_min])).max() 
    
    ################################
    def clean_infeasible_set(self , tol = 1):
        """
        Set default policy and cost2go to cf.INF for state for  which it is unavoidable
        that they will reach unallowable regions

        """
        
        pass
    
    
    
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """

    
    from pyro.dynamic  import pendulum
    from pyro.planning import discretizer
    from pyro.analysis import costfunction
    from functionapproximation import QuadraticFunctionApproximator
    from functionapproximation import MultipleGaussianFunctionApproximator

    sys  = pendulum.SinglePendulum()

    # Discrete world 
    grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [3] )

    # Cost Function
    qcf = costfunction.QuadraticCostFunction.from_sys(sys)

    qcf.xbar = np.array([ -3.14 , 0 ]) # target
    qcf.INF  = 300
    
    # Approx
    
    fa = QuadraticFunctionApproximator( sys.n , x0 = qcf.xbar )
    
    # Discrete world 
    grid_sys_gaussian = discretizer.GridDynamicSystem( sys , [11,11] , [3] , 0.05)
    X0 = grid_sys_gaussian.state_from_node_id

    #fa = MultipleGaussianFunctionApproximator( X0 , 3.0 ) + fa
    
    fa = MultipleGaussianFunctionApproximator( X0 , 1.0 )

    # DP algo
    dp = LinearApproximateDynamicProgramming( grid_sys, qcf, fa )
    
    dp.alpha = 0.8
    dp.gamma = 1.0
    
    dp.w = dp.w + 20
    
    #dp.solve_bellman_equation( tol = 0.1 )
    dp.compute_steps(100)
    
    #dp.plot_cost2go()
    #dp.plot_cost2go_3D()
    dp.animate_cost2go()
        
        
    