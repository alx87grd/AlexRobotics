# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:01:07 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import vehicle_steering
###############################################################################

# Vehicule dynamical system
sys = vehicle_steering.KinematicBicyleModel()

# Set default wheel velocity and steering angle
sys.ubar = np.array([2,0.1])
sys.x0   = np.array([0,0,0])

# Plot open-loop behavior
tf = 10
sys.compute_trajectory( tf )
sys.plot_trajectory()

# Animate the simulation
sys.animate_simulation()