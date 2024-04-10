#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import suspension

sys  = suspension.QuarterCarOnRoughTerrain()

sys.k  = 10.0
sys.vx = 10.0


# Simulation and animation
sys.compute_trajectory( 10, 10001, 'euler')
sys.plot_trajectory('xu')
sys.animate_simulation()