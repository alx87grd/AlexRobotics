#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:51:49 2020

@author: alex
"""
import numpy as np

from gro640_robots import DrillingRobot


sys = DrillingRobot()

tf = 3
n  = 1001
sys.compute_trajectory(tf,n)

###################################################
# Overwrite trajectory with your q(t) data
sys.traj.t       = np.linspace(0,tf, n)    # t
sys.traj.x[:,0]  = np.linspace(0,1, n)     # q1(t)
sys.traj.x[:,1]  = np.linspace(1.4,2, n)   # q2(t)
sys.traj.x[:,2]  = np.linspace(-1.3,2, n)  # q3(t)
###################################################

# Visualise trajectory with animation
sys.animate_simulation( is_3d = True )