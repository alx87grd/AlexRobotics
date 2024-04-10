# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:13:40 2018

@author: Alexandre
"""

from pyro.dynamic  import equation

sys      = equation.VanderPol()

sys.x0[1] = 4.0
sys.compute_trajectory( tf = 20 )
sys.plot_phase_plane_trajectory()
#sys.plot_trajectory()

sys.x0[1] = 0.1
sys.compute_trajectory( tf = 20 )
sys.plot_phase_plane_trajectory()