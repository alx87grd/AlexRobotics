# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:17:36 2018

@author: Alexandre
"""

###############################################################################
import numpy as np

###############################################################################
from pyro.dynamic  import integrator
from pyro.analysis import costfunction
    
###################################
# Simple integrator
###################################

sys = integrator.DoubleIntegrator()

sys.ubar = np.array([1]) # constant input = 1


###################################
# Analysis
###################################
    
# Phase plane behavior test
sys.plot_phase_plane()

# Simulation
sys.x0 = np.array([0,0])

traj = sys.compute_trajectory( tf = 10 )

sys.plot_trajectory()
sys.plot_trajectory('y')
sys.plot_phase_plane_trajectory()

# Cost computing

qcf           = costfunction.QuadraticCostFunction(2, 1, 1)
traj_with_qcf = qcf.trajectory_evaluation( traj )
sys.traj      = traj_with_qcf
sys.plot_trajectory('xuj')

tcf           = costfunction.TimeCostFunction( np.array([0,0]) )
traj_with_tcf = tcf.trajectory_evaluation( traj )
sys.traj      = traj_with_tcf
sys.plot_trajectory('xuj')