# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:34:44 2019

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.control  import robotcontrollers
from pyro.control  import nonlinear
from pyro.dynamic  import manipulator
###############################################################################

sys     = manipulator.TwoLinkManipulator()

# Estimated parameters
sys.l1      = 0.3
sys.l2      = 0.3
sys.lc1     = 0.3
sys.lc2     = 0.3
sys.I1      = 0.5
sys.I2      = 0.2
sys.m1      = 0.9
sys.m2      = 0.05
sys.u_lb[0] = -1.0
sys.u_lb[1] = -1.0
sys.u_ub[0] = 1.0
sys.u_ub[1] = 1.0
sys.l_domain = 0.6

# Target
q_desired = np.array([0.,0.])

# Joint PD
dof = 2
joint_pd      = robotcontrollers.JointPD( dof )
joint_pd.rbar = q_desired
joint_pd.kp   = np.array([ 3.0, 3.0 ])
joint_pd.kd   = np.array([ 1.0, 1.0 ])

# Effector PD
eff_pd      = robotcontrollers.EndEffectorPD( sys )
eff_pd.rbar = np.array([-0.4,+0.2])
eff_pd.kp   = np.array([ 3.0, 3.0 ])
eff_pd.kd   = np.array([ 1.0, 1.0 ])

## Computed torque controller
ct_ctl      = nonlinear.ComputedTorqueController( sys )
ct_ctl.w0   = 2.0
ct_ctl.zeta = 0.7
ct_ctl.rbar = np.array([0.0,0.0])

# Controller selection
ctl = joint_pd
ctl = ct_ctl
ctl = eff_pd

# Closed-loops
cl_sys    = ctl + sys

# Simulations
tf = 5
cl_sys.x0 = np.array([-3.14,0,0,0])
cl_sys.compute_trajectory( tf )
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()
