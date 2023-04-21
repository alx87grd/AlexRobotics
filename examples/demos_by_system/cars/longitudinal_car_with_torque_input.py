#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:31:41 2022

@author: alex
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import longitudinal_vehicule
####################################

sys  = longitudinal_vehicule.LongitudinalFrontWheelDriveCarWithTorqueInput()


sys.x0      = np.array([0,0.1,0,0])
sys.ubar[0] = 2000


sys.compute_trajectory(10,10001,'euler')

sys.plot_trajectory('xy')
sys.animate_simulation( time_factor_video = 1 )