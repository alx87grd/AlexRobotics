# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:34:44 2019

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.planning import randomtree
from pyro.dynamic  import manipulator
from pyro.analysis import simulation
from pyro.analysis import graphical
###############################################################################

torque_controlled_robot = manipulator.FiveLinkPlanarManipulatorwithObstacles()
speed_controlled_robot  = manipulator.SpeedControlledManipulator.from_manipulator( torque_controlled_robot )


traj = simulation.Trajectory.load( 'fivelinkplan.npy' )

animator = graphical.Animator( speed_controlled_robot )

animator.animate_simulation( traj )
