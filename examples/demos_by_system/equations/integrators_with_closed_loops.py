# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:24:51 2018

@author: Alexandre
"""

import numpy as np
############################################################################
from pyro.dynamic  import integrator
from pyro.control  import linear
############################################################################

# Double integrator
si = integrator.SimpleIntegrator()
di = integrator.DoubleIntegrator()
ti = integrator.TripleIntegrator()

# Controller 
ctl      = linear.ProportionalController()

ctl.K[0,0] = 1.0

# New cl-dynamic
clsi = ctl + si

clsi.x0 = np.array([10])
clsi.plot_trajectory('xu')
clsi.plot_phase_plane_trajectory(0, 0)

cldi = ctl + di

cldi.x0 = np.array([10, 0])
cldi.plot_trajectory('xu')
cldi.plot_phase_plane_trajectory(0, 1)

clti = ctl + ti

clti.x0 = np.array([0.1,0,0])
clti.plot_trajectory('xu')
clti.plot_phase_plane_trajectory_3d()