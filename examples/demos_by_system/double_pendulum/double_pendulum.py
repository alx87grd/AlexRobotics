# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""

#import matplotlib
#matplotlib.use('Qt4Agg')

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import pendulum
###############################################################################


sys = pendulum.DoublePendulum()

# Simultation
sys.x0  = np.array([-0.1,0,0,0])
sys.plot_trajectory()
sys.plot_phase_plane_trajectory(0, 2)
sys.animate_simulation()
