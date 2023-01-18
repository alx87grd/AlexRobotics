# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import cartpole
###############################################################################

sys  = cartpole.RotatingCartPole()

# Simultation
sys.x0 = np.array([0,0.1,0,0])
sys.compute_trajectory()
sys.plot_phase_plane_trajectory()
sys.plot_trajectory('xu')
sys.animate_simulation(time_factor_video=1.0, is_3d=True)