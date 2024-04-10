# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:01:07 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
from pyro.dynamic import vehicle_steering
###############################################################################
import advanced_vehicles
import test_vehicle_controllers
###############################################################################

# "Fake" controller - Varying inputs (delta, T_f, T_r) throughout time (change in linear.py)
ctl = test_vehicle_controllers.kinematicInputs()
# Vehicule dynamical system
sys = vehicle_steering.KinematicBicyleModel()

# Set default wheel velocity and steering angle
cl_sys = ctl+sys


# Plot open-loop behavior (ex: np.array[intial_conditions], time_of_simulation)
cl_sys.x0 = np.array([0,0,0])
cl_sys.plot_trajectory()

# Rebuild x,u and t from simulation
x = cl_sys.traj.x
u = cl_sys.traj.u
t = cl_sys.traj.t   

# Plot the vehicle's trajectory
figsize   = (7, 4)
dpi       = 100
plt.figure(2,figsize=figsize, dpi=dpi)
plt.axes(xlim=(-20,150), ylim=(-5,200))  
plt.plot(x[:,0],x[:,1],'--g', label = 'Kinematic Model')     
plt.legend(fontsize ='15')   
plt.legend(fontsize ='15')
plt.title("Vehicle's cartesian position", fontsize=20)
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.show()

# Animate the simulation
cl_sys.animate_simulation()