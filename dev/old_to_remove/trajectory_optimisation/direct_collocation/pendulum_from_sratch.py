#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:12:11 2021

@author: alexandregirard
"""

import numpy as np
from scipy.optimize import minimize

from pyro.dynamic  import pendulum



sys = pendulum.SinglePendulum()

n = 2
m = 1
grid = 30
dt   = 0.2


#dec = np.linspace(0,grid*3,grid*3)


def dec2xu(dec):
    
    x = np.zeros((n,grid))
    u = np.zeros((m,grid))
    
    x[0,:] = dec[0:grid]
    x[1,:] = dec[grid:2*grid]
    u[0,:] = dec[2*grid:3*grid]
    
    return x,u

def cost(dec):
    
    J = 0
    
    x = np.zeros((n,grid))
    u = np.zeros((m,grid))
    
    x[0,:] = dec[0:grid]
    x[1,:] = dec[grid:2*grid]
    u[0,:] = dec[2*grid:3*grid]
    
    for i in range(grid-1):
        
        target = -3.14
        J = J + 0.5 * dt * ( (x[0,i]-target)**2 + (x[0,i+1]-target)**2 + x[1,i]**2 + x[1,i+1]**2 + u[0,i]**2 + u[0,i+1]**2 )
        
        #J = J + 0.5 * dt * ( u[0,i]**2 + u[0,i+1]**2 ) 
        
    return J

def constraints(dec):
    
    x,u = dec2xu(dec)
    
    vec=np.zeros(grid*2-2)
    
    for i in range(grid-1):
        
        vec[i]      = (x[0,i+1] - x[0,i]) - 0.5 * dt * ( sys.f(x[:,i],u[:,i])[0] + sys.f(x[:,i+1],u[:,i+1])[0] )
        vec[i+grid-1] = (x[1,i+1] - x[1,i]) - 0.5 * dt * (  sys.f(x[:,i],u[:,i])[1] + sys.f(x[:,i+1],u[:,i+1])[1] )
    return vec


def compute_bounds():
    bounds = []
    
    bounds.append( (0,0.01) )
    
    for i in range(1,grid-1):
        
        bounds.append( (-4,1) )
        
    bounds.append( (-3.14,-3.13) )
        
    bounds.append( (0,0.01) )
    
    for i in range(1,grid-1):
        
        bounds.append( (-6,4) )
        
    bounds.append( (0,0.01) )
        
    for i in range(grid):
        
        bounds.append( (-10,10) )
        
    return bounds


def display_callback(a):
    
    
    print('Iteration completed')
    
    return True



# Guess
dec = np.zeros(grid*(n+m))
#dec[0:grid] = traj.x[:,0]
#dec[grid:2*grid] = traj.x[:,1]
#dec[2*grid:3*grid] = traj.u[:,0]

bnds = compute_bounds()

cons = {'type': 'eq', 'fun': constraints }
res = minimize( cost, dec, method='SLSQP',  bounds=bnds, constraints=cons, callback=display_callback, options={'disp':True,'maxiter':1000}) #


sys.compute_trajectory(grid*dt,grid)


dec = res.x
    
sys.traj.x[:,0] = dec[0:grid]
sys.traj.x[:,1] = dec[grid:2*grid]
sys.traj.u[:,0] = dec[2*grid:3*grid]


sys.plot_trajectory('xu')
sys.animate_simulation()