# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""


###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import pendulum
###############################################################################


sys = pendulum.DoublePendulum()

# Linearization point
sys.xbar  = np.array([-3.14,0,0,0])

sys.animate_linearized_mode( 0 )
sys.animate_linearized_mode( 2 )