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
###############################################################################

# Vehicule dynamical system
#sys = vehicle.KinematicCarModel()
sys = vehicle_steering.KinematicCarModelwithObstacles()

# Set default wheel velocity and steering angle
sys.ubar = np.array([2,0.2])
sys.x0   = np.array([0,0,0])

# Plot open-loop behavior
tf = 100
sys.compute_trajectory( tf )
sys.plot_trajectory()

# Animate the simulation
sys.animate_simulation( time_factor_video = 5 )