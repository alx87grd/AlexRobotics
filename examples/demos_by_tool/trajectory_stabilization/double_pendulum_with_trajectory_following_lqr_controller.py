# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import pendulum
from pyro.planning import plan
from pyro.analysis import simulation
from pyro.control  import lqr
###############################################################################

sys  = pendulum.DoublePendulum()

traj = simulation.Trajectory.load('double_pendulum_directcollocation_hires.npy')

ctl = lqr.TrajectoryLQRController( sys , traj )

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0 = traj.x[0,:] + np.array([-0.2,0.1,0.1,-0.02])
cl_sys.compute_trajectory(10)
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()