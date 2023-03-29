#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:41:59 2023

@author: alex
"""

import numpy as np

###############################################################################
### Function approximator
###############################################################################

class FunctionApproximator():

    ############################
    def __init__(self, n = 10 ):
        
        self.n = n # number of parameters
        
        #self.w = np.zeros( self.n )
        
    
    ############################
    def J_hat( self , x , w ):
        """ return approx at a state x for given param w """
        
        return 0
    
    
    ############################
    def dJ_dw( self , x ):
        """ return approx a state x """
        
        return np.zeros( self.n )
    

###############################################################################
### Linear Function approximator
###############################################################################
    
class LinearFunctionApproximator( FunctionApproximator ):

    ############################
    def __init__(self, n = 10 ):
        
        FunctionApproximator.__init__(self,n)
        
    
    ############################
    def J_hat( self , x , w ):
        """ return approx a state x """
        
        J_hat = w.T @ self.compute_kernel( x )
        
        return J_hat
    
    
    ############################
    def dJ_dw( self , x ):
        """ return approx a state x """
        
        return self.compute_kernel()
    
    
    ############################
    def kernel( self , x , i ):
        
        # placer holder
        
        return 0
    
    
    ############################
    def compute_kernel( self , x ):
        """ return approx a state x """
        
        phi = np.zeros( self.n )
        
        for i in range(self.n):
            
            phi[i] = self.kernel( x, i )
            
        return phi
    
    
    ############################
    def least_square_fit( self , Js , Xs ):
        """ solve J_d = P w """
        
        n = Js.shape[0]
        
        P = np.zeros((n,self.n,))
        
        for i in range(n):
            
            P[i,:] = self.compute_kernel( Xs[i,:] )
            
        w = np.linalg.lstsq( P , Js , rcond=None)[0]
        
        # DEV
        self.P = P
            
        return w
    
    
    ############################
    def test_Jhat( self , w ):
        """ solve J_d = P w """
        
        J_hat = self.P @ w
            
        return J_hat
    


###############################################################################
### Quadratic Function approximator
###############################################################################
    
class QuadraticFunctionApproximator( LinearFunctionApproximator ):

    ############################
    def __init__(self, sys_n = 2 , x0 = None):
        """
        J_hat = C + B x + x' A x = w' phi

        """
        
        self.sys_n = sys_n
        
        if x0 is not None:
            
            self.x0 = x0
            
        else:
            
            self.x0 = np.zeros( sys_n )
        
        self.n_2_diag = sys_n
        self.n_2_off  = int((sys_n**2-sys_n)/2)
        self.n_2      = +self.n_2_diag + self.n_2_off # 2nd order number of weight
        self.n_1      = sys_n                    # 1nd order number of weight
        self.n_0      = 1                        # 0nd order number ofweight
        
        # Total number of parameters
        self.n = int(self.n_2 + self.n_1 + self.n_0)
    
    
    ############################
    def compute_kernel( self , x ):
        """ return approx a state x """
        
        phi = np.zeros( self.n )
        
        xxT = np.outer( x , x )
        
        #indices
        n0 = self.n_0
        n1 = self.n_0 + self.n_1
        n2 = self.n_0 + self.n_1 + self.n_2_diag
        n3 = self.n_0 + self.n_1 + self.n_2_diag + self.n_2_off
        
        phi[0]     = 1
        phi[n0:n1] = x
        phi[n1:n2] = np.diag( xxT )
        phi[n2:n3] = xxT[np.triu_indices( self.sys_n, k = 1)]
            
        return phi



###############################################################################
### Main
###############################################################################


if __name__ == "__main__":     
    """ MAIN TEST """
    
    from pyro.dynamic  import pendulum
    from pyro.planning import discretizer
    from pyro.analysis import costfunction
    from pyro.planning import dynamicprogramming 

    sys  = pendulum.SinglePendulum()

    # Discrete world 
    grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [3] )

    # Cost Function
    qcf = costfunction.QuadraticCostFunction.from_sys(sys)

    qcf.xbar = np.array([ -3.14 , 0 ]) # target
    qcf.INF  = 300

    # DP algo
    dp = dynamicprogramming.DynamicProgrammingWithLookUpTable(grid_sys, qcf)
    
    dp.solve_bellman_equation( tol = 1.0 )

    
    qfa = QuadraticFunctionApproximator( sys.n , x0 = qcf.xbar )
    
    
    w = qfa.least_square_fit( dp.J , grid_sys.state_from_node_id )
    
    dp.plot_cost2go()
    
    J_hat = qfa.test_Jhat( w )
    
    dp.J = J_hat
    
    dp.plot_cost2go()
    
            
            