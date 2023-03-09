#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:25:38 2020

@author: alex
"""

import numpy as np
from pyro.dynamic  import pendulum
from pyro.control  import nonlinear
from pyro.analysis import simulation

sys = pendulum.DoublePendulum()

sys.x0 = np.array([0.0,1.,1.,0.])
sys.compute_trajectory()

