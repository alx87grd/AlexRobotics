# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""
##############################################################################
import numpy as np
##############################################################################


##############################################################################
# Dynamics
##############################################################################

from pyro.dynamic  import pendulum

# Define the dynamical system to control
sys  = pendulum.SinglePendulum()

sys.plot_phase_plane()

#Max/Min torque
sys.u_ub[0] = +5
sys.u_lb[0] = -5

sys.xbar  = np.array([ -3.14 , 0 ]) # target state [ position , velocity]
sys.x0    = np.array([ -0.00 , 0 ]) # initial state

##############################################################################
# Cost Function
##############################################################################

from pyro.analysis import costfunction

cf = costfunction.QuadraticCostFunction.from_sys( sys ) 

cf.INF  = 10000     # The value iteration algo needs this parameter

cf.Q[0,0] = 1
cf.Q[1,1] = 0.1
cf.R[0,0] = 1

print('Q=\n',cf.Q)
print('R=\n',cf.R)

sys.cost_function = cf

##############################################################################
# Sub-optimal PID
##############################################################################

from pyro.control  import linear

pid      = linear.ProportionalController(1,2) # 1 output from 2 inputs
pid.rbar = sys.xbar                           # target
pid.K    = np.array([10,4])                   # gain matrix

print('PID K=\n',pid.K)
pid.plot_control_law( sys = sys )

sys_with_pid = pid + sys
sys_with_pid.plot_trajectory('xuj')
sys_with_pid.animate_simulation()

print('Trajectory cost: ', sys_with_pid.traj.J[-1])

##############################################################################
# LQR
##############################################################################

from pyro.dynamic  import statespace
from pyro.control  import lqr

# Linear model
ss  = statespace.linearize( sys )

print('A=\n',ss.A)
print('B=\n',ss.B)

lqr_ctl = lqr.synthesize_lqr_controller( ss , cf , sys.xbar)

print('LQR K=\n',lqr_ctl.K)
lqr_ctl.plot_control_law( sys = sys )

sys_with_lqr = lqr_ctl + sys
sys_with_lqr.plot_trajectory('xuj')
sys_with_lqr.animate_simulation()

print('Trajectory cost: ', sys_with_lqr.traj.J[-1])

##############################################################################
# VI
##############################################################################

from pyro.planning import valueiteration
from pyro.planning import discretizer

# Value iteration algo
vi = valueiteration.ValueIteration_2D( discretizer.GridDynamicSystem( sys ) , cf )

vi.initialize()
#vi.compute_steps(200,True). # To compute from sratch instead of loading the solution
vi.load_data('simple_pendulum_vi') # /content/pyro/examples/demo/
vi.assign_interpol_controller()

vi_ctl = vi.ctl

vi_ctl.plot_control_law( sys = sys , n=1000)

sys_with_vi = vi_ctl + sys
sys_with_vi.plot_trajectory('xuj')
sys_with_vi.animate_simulation()

print('Trajectory cost: ', sys_with_vi.traj.J[-1])

##############################################################################
# Direct Collocation Trajectory Optimisation
##############################################################################

from pyro.planning import trajectoryoptimisation

planner = trajectoryoptimisation.DirectCollocationTrajectoryOptimisation( sys ,
                                                                         dt = 0.1,
                                                                         grid = 50)

planner.x_start = sys.x0
planner.x_goal  = sys.xbar

planner.compute_optimal_trajectory()
planner.show_solution()

##############################################################################
# Stabilizing the optimal trajectory
##############################################################################

from pyro.control  import nonlinear

traj_ctl  = nonlinear.ComputedTorqueController( sys , planner.traj)  

traj_ctl.w0   = 2
traj_ctl.zeta = 1
traj_ctl.rbar = sys.xbar[0:1]

traj_ctl.plot_control_law( sys = sys , n=100)


sys_with_traj_ctl = traj_ctl + sys
sys_with_traj_ctl.plot_trajectory('xuj')
sys_with_traj_ctl.animate_simulation()

print('Trajectory cost: ', sys_with_traj_ctl.traj.J[-1])

