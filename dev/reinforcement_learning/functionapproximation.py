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
        """ Compute J approx given state x and param w """
        
        return 0
    
    
    ############################
    def dJ_dw( self , x ):
        """ Compute dJ/dw given state x """
        
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
        """ Compute J approx given state x and param w """
        
        phi   = self.compute_kernel( x )
        
        J_hat = phi.T @ w
        
        return J_hat
    
    
    ############################
    def dJ_dw( self , x ):
        """ Compute dJ/dw given state x """
        
        phi   = self.compute_kernel( x )
        
        return phi
    
    
    ############################
    def compute_kernel( self , x ):
        """ Compute kernel functions phi given a state x """
        
        phi = np.zeros( self.n )
            
        return phi
    
    
    ############################
    def compute_all_kernel( self , Xs ):
        """ Compute all kernel functions phi given a m state x """
        
        m = Xs.shape[0]   # number of data point
        n = self.n        # number of params
        
        P = np.zeros( ( m , n ))
        
        for i in range(m):
            
            P[i,:] = self.compute_kernel( Xs[i,:] )
            
        return P.T
    
    
    ############################
    def least_square_fit( self , Js , P ):
        """ solve J_d = P w """

        #P = self.compute_all_kernel( Xs )
            
        w = np.linalg.lstsq( P.T , Js , rcond=None)[0]
        
        J_hat = P.T @ w
            
        return w , J_hat
    
    #############################
    def __add__(self, other ):
        """ 
        return new function approx with all kernels
        """
        
        return CombinedLinearFunctionApproximator( self , other )
    
    
    
###############################################################################
### Combined Linear Function approximator
###############################################################################
    
class CombinedLinearFunctionApproximator( LinearFunctionApproximator ):

    ############################
    def __init__(self, FA1 , FA2 ):
        
        FunctionApproximator.__init__(self, FA1.n + FA2.n )
        
        self.FA1 = FA1
        self.FA2 = FA2
        
        
    
    ############################
    def compute_kernel( self , x ):
        """ Compute kernel functions phi given a state x """
        
        phi1 = self.FA1.compute_kernel( x )
        phi2 = self.FA2.compute_kernel( x )
        
        phi = np.concatenate(( phi1, phi2 ))
            
        return phi
        
        
    
    


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
        
        x = x - self.x0
        
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
### Radial basis function Function approximator
###############################################################################
    
class GaussianFunctionApproximator( LinearFunctionApproximator ):

    ############################
    def __init__(self, x0 , sig = 6.0 ):
        """
        J_hat = exp( - || x - x0 || / 2 sig^2 )

        """
        
        self.x0    = x0
        self.sys_n = x0.shape[0]
        self.n     = 1
        self.a     = -0.5  / (sig**2)
    
    
    ############################
    def compute_kernel( self , x ):
        """ return approx a state x """
        
        phi = np.array([0.])
        
        e = x - self.x0
        r = e.T @ e
        
        phi[0] = np.exp( self.a * r )
            
        return phi
    
    
###############################################################################
class MultipleGaussianFunctionApproximator( LinearFunctionApproximator ):

    ############################
    def __init__(self, Xs , sig = 1.0 ):
        """
        J_hat = sum exp( - || x - x0 || / 2 sig^2 )

        """
        
        self.Xs    = Xs
        self.sys_n = Xs.shape[1]
        self.n     = Xs.shape[0]   # number of data point
        self.a     = -0.5  / (sig**2)
    
    
    ############################
    def compute_kernel( self , x ):
        """ return approx a state x """
        
        phi = np.zeros(self.n)
        
        for i in range(self.n):
            
            e = x - self.Xs[i,:] 
            r = e.T @ e
            
            phi[i] = np.exp( self.a * r )
            
        return phi



###############################################################################
### Main
###############################################################################


if __name__ == "__main__":     
    """ MAIN TEST """
    
    from pyro.dynamic  import pendulum
    from pyro.dynamic  import massspringdamper
    from pyro.planning import discretizer
    from pyro.analysis import costfunction
    from pyro.planning import dynamicprogramming 

    sys  = pendulum.SinglePendulum()
    #sys  = 

    # Discrete world 
    grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [3] )

    # Cost Function
    qcf = costfunction.QuadraticCostFunction.from_sys(sys)

    qcf.xbar = np.array([ -3.14 , 0 ]) # target
    qcf.INF  = 200

    # DP algo
    dp = dynamicprogramming.DynamicProgrammingWithLookUpTable(grid_sys, qcf)
    
    dp.solve_bellman_equation( tol = 1.0 )
    dp.plot_cost2go_3D()
    
    
    
    # Approx
    qfa0 = QuadraticFunctionApproximator( sys.n , x0 = qcf.xbar )
    
    sig = 2.0
    
    qfa1 = GaussianFunctionApproximator( x0 = np.array([0,0]), sig = sig )
    qfa2 = GaussianFunctionApproximator(  x0 = qcf.xbar , sig = sig)
    qfa3 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([1,0]) , sig = sig)
    qfa4 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([1,1]) , sig = sig)
    qfa5 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([0,1]) , sig = sig)
    qfa6 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([-1,0]) , sig = sig)
    qfa7 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([-1,-1]) , sig = sig)
    qfa8 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([0,-1]) , sig = sig)
    qfa9 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([0.5,0]) , sig = sig)
    qfa10 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([0.5,0.5]) , sig = sig)
    qfa11 = GaussianFunctionApproximator(  x0 = qcf.xbar + np.array([0,0.5]) , sig = sig)
    qfa = qfa0 + qfa1 + qfa2 + qfa3 + qfa4 + qfa5 + qfa6 + qfa7 + qfa8 + qfa9 + qfa10 + qfa11
    
    Xs = grid_sys.state_from_node_id # All state on the grid
    
    P = qfa.compute_all_kernel( Xs )
    
    w , J_hat = qfa.least_square_fit( dp.J , P )
    
    grid_sys.plot_grid_value_3D(  J_hat , None , ' J_hat')
    grid_sys.plot_grid_value_3D( dp.J , J_hat , 'J vs. J_hat')
    
            
            