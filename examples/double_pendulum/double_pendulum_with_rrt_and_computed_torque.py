# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import pendulum
from pyro.control  import nonlinear
from pyro.planning import randomtree

###############################################################################


sys  = pendulum.DoublePendulum()

###############################################################################

x_start = np.array([-3.14,0,0,0])
x_goal  = np.array([0,0,0,0])

planner = randomtree.RRT( sys , x_start )

t = 12
    
planner.u_options = [
        np.array([-t,-t]),
        np.array([-t,+t]),
        np.array([+t,-t]),
        np.array([+t,+t]),
        np.array([ 0,+t]),
         np.array([ 0,-t]),
        np.array([ 0, 0]),
        np.array([+t, 0]),
        np.array([-t, 0])
        ]

planner.goal_radius          = 0.8
planner.dt                   = 0.1
planner.max_nodes            = 12000
planner.max_solution_time    = 8
planner.max_distance_compute = 500

planner.dyna_plot            = False

planner.find_path_to_goal( x_goal )

planner.plot_tree()

###############################################################################

# Controller

traj = planner.trajectory

ctl  = nonlinear.ComputedTorqueController( sys , traj )

ctl.w0   = 1.0
ctl.zeta = 0.7

# goal
ctl.rbar = np.array([0,0])


# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0 = x_start
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()