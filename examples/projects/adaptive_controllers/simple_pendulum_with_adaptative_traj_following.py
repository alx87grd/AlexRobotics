# -*- coding: utf-8 -*-
"""
Created on 19/11/2019

@author: Pierre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import pendulum
from adaptive_computed_torque import SinglePendulumAdaptativeController
###############################################################################

sys = pendulum.SinglePendulum()


sys.m1       = 1
sys.l1       = 2
sys.lc1      = 1
sys.I1       = 0
sys.gravity  = 9.81 


# Controller

ctl = SinglePendulumAdaptativeController(sys)

# Sys ID initial guess
m1_hat = 0
l1_hat = 0

ctl.z0[0] = m1_hat * l1_hat ** 2
ctl.z0[1] = m1_hat * l1_hat * 9.81

def t2traj( t , r ):
    """  """
    a = 1.5
    w = 2.5
    v = 1.0
    
    ddq_d          =   - a * w * w * np.sin( w * t)
    dq_d           =   a * w * np.cos( w * t) + v
    q_d            =   r + a * np.sin( w * t) + v * t
    
    return [ ddq_d , dq_d , q_d ] 



# Controller Param
ctl.K      = 10
ctl.lam    = 2
ctl.P[0,0] = 10
ctl.P[1,1] = 30
ctl.get_traj = t2traj


# Set Point
q_target = np.array([0.0]) # np.array([3.14])
ctl.rbar = q_target

# New cl-dynamic
cl_sys = ctl + sys

cl_sys.state_label[2] = 'H'
cl_sys.state_label[3] = 'mgl'


# Simultation
cl_sys.x0[0]  = 0

cl_sys.compute_trajectory(tf=30, n=20001) # solver='euler')
cl_sys.plot_phase_plane_trajectory()
cl_sys.plot_trajectory_with_internal_states()
cl_sys.animate_simulation()