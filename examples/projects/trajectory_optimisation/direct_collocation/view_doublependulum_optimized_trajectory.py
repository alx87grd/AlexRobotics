#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:11:12 2021

@author: alex
"""
from pyro.dynamic  import pendulum
from pyro.analysis import Trajectory

sys = pendulum.DoublePendulum()

#sys.traj = Trajectory.load('doublependulumswingup2sec.npy')
sys.traj = Trajectory.load('doublependulumswingup4sec.npy')

sys.plot_trajectory('xu')
sys.animate_simulation()