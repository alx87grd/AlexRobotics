# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import cartpole
from pyro.planning import randomtree
from pyro.planning import trajectoryoptimisation 
###############################################################################


sys  = cartpole.UnderActuatedRotatingCartPole()

sys.u_ub[0] = +50 # Max torque
sys.u_lb[0] = -50 # Min torque

###############################################################################

x_start = np.array([0,-3.14,0,0])
x_goal  = np.array([0,0,0,0])

rrt = randomtree.RRT( sys , x_start )
    
rrt.u_options = [ sys.u_ub , sys.u_lb ]

rrt.goal_radius          = 1.5
rrt.dt                   = 0.1
rrt.max_nodes            = 10000
rrt.max_solution_time    = 3.0
rrt.max_distance_compute = 1000
rrt.dyna_plot            = False

rrt.find_path_to_goal( x_goal )

###############################################################################

planner = trajectoryoptimisation.DirectCollocationTrajectoryOptimisation( sys , 0.1 , 30 )

planner.x_start = x_start
planner.x_goal  = x_goal

planner.set_initial_trajectory_guest( rrt.traj )

planner.maxiter = 500
planner.compute_optimal_trajectory()
planner.show_solution()
planner.animate_solution( is_3d = True )

