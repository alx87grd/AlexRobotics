# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:09:37 2017

@author: Charles.Khazoom@hotmail.com
"""

import numpy as np
import numpy.matlib as mb
import matplotlib.pyplot as plt

from scipy.optimize import minimize, NonlinearConstraint, rosen, rosen_der,HessianUpdateStrategy,BFGS#, interpolate
from scipy.interpolate import interp1d
from pyro.dynamic import manipulator
from pyro.planning import plan
from pyro.analysis import costfunction
from pyro.analysis import Trajectory

from pyro.control  import nonlinear
from pyro.analysis import simulation

'''
##############################################################################
'''


#class TorqueSquaredCostFunction( costfunction.QuadraticCostFunction ):
#    def __init__(self):
#        costfunction.QuadraticCostFunction.__init__( self )

'''
##############################################################################
'''
# create system 
#sys = manipulator.OneLinkManipulator()

from pyro.dynamic  import pendulum

sys  = pendulum.DoublePendulum()

'''
Create optization problem
'''

# set cost function
# minimize torque square is quadraic cost with Q and V set to 0
#class TorqueSquaredCostFunction( costfunction.QuadraticCostFunction ): should implement this class
#sys.cost_function.Q=np.zeros(sys.cost_function.Q.shape)
#sys.cost_function.V=np.zeros(sys.cost_function.V.shape)
sys.cost_function.Q[0,0] = 1
sys.cost_function.Q[1,1] = 1
sys.cost_function.R[0,0] = 1
sys.cost_function.R[1,1] = 1
sys.cost_function.xbar = np.array([0,0,0,0])

global ngrid
ngrid = 30 # number of gridpoint

######## set bounds ##########
ub_t0 = 0 # bounds on t0 
lb_t0 = 0

ub_tf = 8 # bounds on tf 
lb_tf = 7
# should implement an error message if tf is not >t0 and in t0<0

#ub_state = np.array([])
#lb_state
sys.x_ub=[+2*np.pi,+2*np.pi,0,0]
sys.x_lb=[-2*np.pi,-2*np.pi,0,0]

sys.u_ub = [+50.0,+50.0]
sys.u_lb = [-50.0,-50.0]

ub_x = sys.x_ub # bounds on x
lb_x = sys.x_lb

ub_u = sys.u_ub # bounds on inputs u
lb_u = sys.u_lb

ub_x0 = [-3.14,0,0,0] # bounds on inital state
lb_x0 = [-3.14,0,0,0]


ub_xf = [0,0,0,0] #sys.x_ub # bounds on final state
lb_xf = [0,0,0,0]

'''
create initial guess
'''

#x_guess = np.linspace(ub_x0, ub_xf, ngrid)
#u_guess = np.ones([ngrid,sys.m])
#t_guess = np.linspace(ub_t0, ub_tf, ngrid)
#y_guess= x_guess
#dx_guess=np.zeros(x_guess.shape)
#
#for i in range(ngrid):
#    dx_guess[i,] = sys.f(x_guess[i,],u_guess[i,]) # compute f
#
#guess_traj = Trajectory(x_guess, u_guess, t_guess, dx_guess, y_guess)



######## set equality contraints (other than dynamics) ##########
# sys.f contains the dynamics ! 

'''
Convert to non-linear program (direct collocation)

decision variables are ordered:
    
    x0[0],x0[1]...,x0[ngrid-1], x1[0],x1[1]...,x1[ngrid-1],...,...xn[0],xn[1]...,xn[ngrid-1]
    u0[0],u0[1]...,u0[ngrid-1], u1[0],x1[1]...,u1[ngrid-1],...,...un[0],un[1]...,un[ngrid-1],
    t0, tf
'''

#convert bounds in discete form, for all x(.),u(.),t(.)

def pack_bounds(lb_x,ub_x,lb_x0,ub_x0,lb_xf,ub_xf,lb_u,ub_u,lb_t0,ub_t0,lb_tf,ub_tf):
    
    bnds = np.array([]).reshape(0,2) # initialize bounds np.array to append to
    
    for i in range(sys.n): # convert bounds on x
        bnd_arr_to_add = np.concatenate([mb.repmat(lb_x[i], ngrid, 1),mb.repmat(ub_x[i], ngrid, 1)],axis=1)
        bnd_arr_to_add[0,:]=np.array([lb_x0[i],ub_x0[i]])#enforce bound on initial value
        bnd_arr_to_add[ngrid-1,:]=np.array([lb_xf[i],ub_xf[i]])#enforce bound on final value
        bnds = np.append(bnds,bnd_arr_to_add,axis=0)
    
    for i in range(sys.m): # convert bounds on u
        bnd_arr_to_add = np.concatenate([mb.repmat(lb_u[i], ngrid, 1),mb.repmat(ub_u[i], ngrid, 1)],axis=1)
        bnds = np.append(bnds,bnd_arr_to_add,axis=0)
    
    # append bounds on t0 and tF
    bnds=np.append(bnds,np.array([lb_t0,lb_t0]).reshape(1,2),axis=0)
    bnds=np.append(bnds,np.array([lb_tf,lb_tf]).reshape(1,2),axis=0)
    
    bnds = tuple(map(tuple, bnds))#convert bnds np.array to tuple for input in NLP solver
    
    return bnds


def traj_2_dec_var(traj):
    
    dec_vars = np.array([]).reshape(0,1) # initialize dec_vars array
    
    for i in range(sys.n): # append states x
        arr_to_add = traj.x[:,i].reshape(ngrid,1)
        dec_vars = np.append(dec_vars,arr_to_add,axis=0)
    
    for i in range(sys.m): # append inputs u
        arr_to_add = traj.u[:,i].reshape(ngrid,1)
        dec_vars = np.append(dec_vars,arr_to_add,axis=0)
    
    # append t0 and tF
    dec_vars=np.append(dec_vars,np.array(traj.t[0]).reshape(1,1),axis=0)
    dec_vars=np.append(dec_vars,traj.t[-1].reshape(1,1),axis=0)
    dec_vars=dec_vars.reshape(ngrid*(sys.n+sys.m)+2,)
    return dec_vars




def unpack_dec_var(decision_variables):
    global ngrid 
    x = np.zeros([ngrid,sys.n])
    u = np.zeros([ngrid,sys.m])
    # unpack decision variables into trajectory
    for i in range(0,sys.n):
        x[:,i] = decision_variables[i*ngrid:(i+1)*ngrid].reshape(ngrid)

    for i in range(sys.n,sys.n+sys.m):
        u[:,i-sys.n] = decision_variables[i*ngrid:(i+1)*ngrid].reshape(ngrid)
    
    t0 = decision_variables[-2]
    tf = decision_variables[-1]
    return x,u,t0,tf

def compute_dx(x,u,sys):
    dx = np.zeros(x.shape)
    global ngrid 
    for i in range(ngrid):
        dx[i,:]=sys.f(x[i,:],u[i,:])
    return dx
    

def dec_var_2_traj(decision_variables):
    global ngrid     
    
    x,u,t0,tf = unpack_dec_var(decision_variables)
    
    dx = compute_dx(x,u,sys)
    
    t = np.linspace(t0, tf, ngrid).reshape(ngrid,)
    
    y=x # find proper fct evaluation later from sys object
    
    traj = Trajectory(x, u, t, dx, y)
    
    return traj

def dynamics_cstr(traj):
    
    t0 = traj.t[0]
    
    tf = traj.t[-1]
    
    h = (tf-t0)/(ngrid-1) #step between each grid
    
    dyn_constr = np.diff(traj.x,axis=0) - (traj.dx[0:-1]+traj.dx[1:])*h/2 # this must be = 0
    dyn_constr = dyn_constr.reshape((ngrid-1)*sys.n,)
    return dyn_constr

def compute_cost(decision_variables):
    traj_opt = dec_var_2_traj(decision_variables) #get trajectorty
    traj_opt = sys.cost_function.trajectory_evaluation(traj_opt)#compute cost fcn
    cost = traj_opt.J[-1]# traj_opt.J is the cumulated cost from integral, we take only the last value
    return cost

def interp_traj(traj,ngrid):
        
    f_x = interp1d(traj.t, traj.x, kind='cubic',axis=0)
    f_u = interp1d(traj.t, traj.u, kind='quadratic',axis=0)
    
    t_interp = np.linspace(traj.t[0],traj.t[-1],ngrid)
    x_interp = f_x(t_interp)
    u_interp = f_u(t_interp)
    dx_interp=compute_dx(x_interp,u_interp,sys)
    y_interp = x_interp
    traj_interp = Trajectory(x_interp, u_interp, t_interp, dx_interp, y_interp)
    return traj_interp 






# Guess traj based from RRT solution
loaded_traj = simulation.Trajectory.load('double_pendulum_rrt.npy')
ngrid       = loaded_traj.time_steps * 10


#

bnds = pack_bounds(lb_x,ub_x0,lb_x,ub_x0,lb_xf,ub_xf,lb_u,ub_u,lb_t0,lb_t0,lb_tf,lb_tf)

cons_slsqp = ({'type': 'eq', 'fun': lambda x: dynamics_cstr(dec_var_2_traj( x ))  })

cons_trust=NonlinearConstraint(lambda x: dynamics_cstr(dec_var_2_traj( x )), 0, 0)#


guess_traj    = interp_traj( loaded_traj , ngrid )
dec_var_guess = traj_2_dec_var( guess_traj )



'''
Solve non-linear program
'''
#method='trust-constr'
res4 = minimize(compute_cost, dec_var_guess,method='SLSQP' , bounds=bnds, constraints=cons_slsqp,tol=1e-6,options={'disp': True,'maxiter':1000})



#dec_var_guess = np.load('C:\Users\Charles Khazoom\Documents\git\pyro\trajresults\solution_with_trust_const.npy')

#res4 = minimize(compute_cost, dec_var_guess,method='trust-constr' , bounds=bnds, constraints=cons_trust,tol=1e-6,options={'disp': True,'maxiter':1000},jac='2-point',hess=BFGS())
# 



'''
Analyze results
'''


result_traj = dec_var_2_traj(res4.x)

result_traj.save('double_pendulum_rrt_optimised.npy')


#show solution vs initial guess
sys.traj = loaded_traj
sys.plot_trajectory('xu')
sys.traj = result_traj
sys.plot_trajectory('xu')
#sys.animate_simulation()

# CLosed-loop controller
ctl = nonlinear.ComputedTorqueController( sys , result_traj )
ctl.rbar = np.array([0,0])

## New cl-dynamic
cl_sys = ctl + sys

cl_sys.x0 = np.array([-3.14,0,0,0])
cl_sys.compute_trajectory( 10 )
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()


print(sys.cost_function.trajectory_evaluation( guess_traj ).J[-1])
print(sys.cost_function.trajectory_evaluation( result_traj ).J[-1])
print(sys.cost_function.trajectory_evaluation( cl_sys.traj ).J[-1])