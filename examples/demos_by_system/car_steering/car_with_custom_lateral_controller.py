#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:16:59 2019

@author: alex
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import vehicle_steering
from pyro.control  import controller
###############################################################################

# Vehicule dynamical system
sys = vehicle_steering.ConstantSpeedKinematicCarModel()


class CarController( controller.StaticController ) :

    ############################
    def __init__(self):
        """ """
        
        controller.StaticController.__init__( self, 1, 1, 3)
        
        # Label
        self.name = 'Car Controller'
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        """
        
        y_desired = r[0]
        
        lenght = 5.0
        speed = 2.0
        x_car = y[0]
        y_car = y[1]
        theta = y[2]
        
        
        # TODO your own feedback law below
        ########################
        delta = 1.5 * ( y_desired - y_car )
        delta = np.clip( delta , -1.5 , 1.5)
        ########################
        
        u = np.array([ delta ])
        
        return u
    

ctl = CarController()
    
cl_sys = ctl + sys




cl_sys.plot_phase_plane(1,2)
#cl_sys.plot_phase_plane_closed_loop(1,2)

cl_sys.x_ub = np.array([+8.,+8.,+8.])
cl_sys.x_lb = np.array([-8.,-8.,-8.])

# Animate the simulation
cl_sys.x0[1] = 0.5
cl_sys.x0[2] = 0.0
cl_sys.plot_trajectory()
anim = cl_sys.animate_simulation()
cl_sys.plot_phase_plane_trajectory(1,2)
#cl_sys.plot_phase_plane_trajectory_closed_loop(1,2)
