# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 20:54:31 2018

@author: Alexandre
"""

import numpy as np

from pyro.dynamic import system


##############################################################################
        
class VanderPol( system.ContinuousDynamicSystem ):
    """ 
    SimpleIntegrator Example for a ContinuousDynamicSystem
    -------------------------------------------------------
    mass driven by a speed input: dx [m/sec] = f(x,u,t) = u [m/sec]
    
    """
    
    ############################
    def __init__(self, mu = 0.5 ):
        """ """
    
        # Dimensions
        self.n = 2   
        self.m = 1   
        self.p = 2
        
        # initialize standard params
        system.ContinuousDynamicSystem.__init__( self, self.n, self.m, self.p)
        
        # Labels
        self.name = 'Van der Pol oscillator'
        
        self.x_ub[0] = 3
        self.x_lb[0] = -3
        self.x_ub[1] = 3
        self.x_lb[1] = -3
        
        self.mu = mu
        
    
    #############################
    def f(self, x  , u  , t = 0 ):
        """ 
        Continuous time foward dynamics evaluation
        
        dx = f(x,u,t)
        
        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1
        
        OUPUTS
        dx : state derivative vectror n x 1
        
        """
        y  = x[0]
        dy = x[1]
        
        
        ddy = - y + self.mu * dy * ( 1 - y**2 )
        
        dx = np.zeros(self.n) # State derivative vector
        
        dx[0] = dy
        dx[1] = ddy
        
        return dx


    
    
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    
    sys = VanderPol()
    
    #sys.plot_phase_plane()
    
    sys.x0[1] = 4.0
    sys.compute_trajectory( tf = 20 )
    sys.plot_phase_plane_trajectory()
    #sys.plot_trajectory()
    
    sys.x0[1] = 0.1
    sys.compute_trajectory( tf = 20 )
    sys.plot_phase_plane_trajectory()