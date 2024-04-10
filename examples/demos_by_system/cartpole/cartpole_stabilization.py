#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:01:30 2023

@author: alex
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import cartpole
from pyro.control import controller
###############################################################################


class CartPoleController( controller.StaticController ) :
    
    ############################
    def __init__( self ):

        super().__init__(1, 1, 4)
        
        # Label
        self.name = 'Cart Pole Controller'
        self.dof = 2
        
        # Gains
        self.kd = 2
        self.kp = 25

    
    #############################
    def c( self , y , r , t = 0 ):
        
        u = np.zeros(self.m) 
        
        #############################
        # State feedback
        q  = y[ 0        :     self.dof   ]
        dq = y[ self.dof : 2 * self.dof   ]
        
        #############################
        # Custom feedback law
        
        u[0] = self.kp * (np.pi - q[1] ) - self.kd * dq[1]
       
        #############################
        # Saturation
        np.clip( u, -50, 50)
        
        return u


sys  = cartpole.CartPole()
ctl  = CartPoleController()

cl_sys = ctl + sys

cl_sys.x0[0] = -3.0
cl_sys.x0[1] = 2.5
    
cl_sys.compute_trajectory( tf = 4.0 )
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video=0.5 )