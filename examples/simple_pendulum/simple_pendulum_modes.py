# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""
##############################################################################
import numpy as np
##############################################################################
from pyro.dynamic  import pendulum
##############################################################################

# Dynamic system
sys  = pendulum.SinglePendulum()

sys.animate_linearized_modes()