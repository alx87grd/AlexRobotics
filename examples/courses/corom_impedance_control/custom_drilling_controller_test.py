#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:51:49 2020

@author: alex
"""
import numpy as np
import matplotlib

from corom_robots import DrillingRobot
from corom_robots import DrillingRobotOnJig
from custom_drilling_controller      import CustomDrillingController


# Model dynamique du robot
sys = DrillingRobotOnJig()

# Controller
model = DrillingRobot()
ctl   = CustomDrillingController( model ) # Empty do nothing controller template

# Closed-loop dynamic
clsys = ctl + sys

# États initiaux
#clsys.x0 =  np.array([0.7,1.45,-1.4,0,0,0]) #  Tombe dans le trou
clsys.x0 =  np.array([0,1.4,-1.3,0,0,0]) #

# Simulation
tf = 6
n = 10001
clsys.compute_trajectory( tf , n)

clsys.plot_trajectory('x')
clsys.plot_trajectory('u')

# Données pour analyse
t        = clsys.traj.t
q_traj   = clsys.traj.x[:,0:3]  # Trajectoire des angles du robot
dq_traj  = clsys.traj.x[:,3:6]  # Trajectoire des vitesses du robot
tau_traj = clsys.traj.u         # Trajectoire des couples du robot

# Force de contact

f_traj   = np.zeros((n,3))
for i in range(n):
    f_traj[i,:] = sys.f_ext( q_traj[i,:] , dq_traj[i,:] )

fig , plots = matplotlib.pyplot.subplots(3)
plots[0].plot( t , f_traj[:,0] )
plots[1].plot( t , f_traj[:,1] )
plots[2].plot( t , f_traj[:,2] )
fig.canvas.manager.set_window_title('Contact forces')



clsys.animate_simulation( is_3d = True )

