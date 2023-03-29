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
    def J( self , x ):
        """ return approx a state x """
        
        return 0
    
    
    ############################
    def dJ_dw( self , x ):
        """ return approx a state x """
        
        return np.zeros( self.n )
    

###############################################################################
### Function approximator
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
        
    
            
            