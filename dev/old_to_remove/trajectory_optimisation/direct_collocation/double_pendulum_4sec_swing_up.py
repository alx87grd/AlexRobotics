#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:12:11 2021

@author: alexandregirard
"""

import numpy as np
from scipy.optimize import minimize

from pyro.dynamic  import pendulum



sys = pendulum.DoublePendulum()

n = 4
m = 2
grid = 20
dt   = 0.2


#dec = np.linspace(0,grid*3,grid*3)


def dec2xu(dec):
    
    x = np.zeros((n,grid))
    u = np.zeros((m,grid))
    
    x[0,:] = dec[0:grid]
    x[1,:] = dec[grid:2*grid]
    x[2,:] = dec[2*grid:3*grid]
    x[3,:] = dec[3*grid:4*grid]
    u[0,:] = dec[4*grid:5*grid]
    u[1,:] = dec[5*grid:6*grid]
    
    return x,u

def cost(dec):
    
    J = 0
    
    x = np.zeros((n,grid))
    u = np.zeros((m,grid))
    
    x[0,:] = dec[0:grid]
    x[1,:] = dec[grid:2*grid]
    x[2,:] = dec[2*grid:3*grid]
    x[3,:] = dec[3*grid:4*grid]
    u[0,:] = dec[4*grid:5*grid]
    u[1,:] = dec[5*grid:6*grid]
    
    for i in range(grid-1):
        
        #J = J + 0.5 * dt * ( (x[0,i]-target)**2 + (x[0,i+1]-target)**2 + x[1,i]**2 + x[1,i+1]**2 + u[0,i]**2 + u[0,i+1]**2 )
        
        J = J + 0.5 * dt * ( x[0,i]**2 + x[0,i+1]**2 + x[1,i]**2 + x[1,i+1]**2 ) 
        J = J + 0.5 * dt * ( x[2,i]**2 + x[2,i+1]**2 + x[3,i]**2 + x[3,i+1]**2 ) 
        J = J + 0.5 * dt * ( u[0,i]**2 + u[0,i+1]**2 + u[1,i]**2 + u[1,i+1]**2 ) * 5
        
    return J

def constraints(dec):
    
    x,u = dec2xu(dec)
    
    vec=np.zeros(grid*4-4)
    
    for i in range(grid-1):
        
        vec[i+grid*0-0] = (x[0,i+1] - x[0,i]) - 0.5 * dt * (  sys.f(x[:,i],u[:,i])[0] + sys.f(x[:,i+1],u[:,i+1])[0] )
        vec[i+grid*1-1] = (x[1,i+1] - x[1,i]) - 0.5 * dt * (  sys.f(x[:,i],u[:,i])[1] + sys.f(x[:,i+1],u[:,i+1])[1] )
        vec[i+grid*2-2] = (x[2,i+1] - x[2,i]) - 0.5 * dt * (  sys.f(x[:,i],u[:,i])[2] + sys.f(x[:,i+1],u[:,i+1])[2] )
        vec[i+grid*3-3] = (x[3,i+1] - x[3,i]) - 0.5 * dt * (  sys.f(x[:,i],u[:,i])[3] + sys.f(x[:,i+1],u[:,i+1])[3] )
        
    return vec


def compute_bounds():
    
    bounds = []
    
    #x0
    bounds.append( (-3.14,-3.13) )
    
    for i in range(1,grid-1):
        
        bounds.append( (-5,2) )
        
    bounds.append( (0.0,0.01) )
    
    #x1
    bounds.append( (0,0.01) )
    
    for i in range(1,grid-1):
        
        bounds.append( (-3,3) )
        
    bounds.append( (0,0.01) )
    
    #x2
    
    bounds.append( (0,0.01) )
    
    for i in range(1,grid-1):
        
        bounds.append( (-3,6) )
        
    bounds.append( (0,0.01) )
    
    #x3
    
    bounds.append( (0,0.01) )
    
    for i in range(1,grid-1):
        
        bounds.append( (-10,10) )
        
    bounds.append( (0,0.01) )
    
    #u0
    
    for i in range(grid):
        
        bounds.append( (-20,30) )
    
    #u1
        
    for i in range(grid):
        
        bounds.append( (-20,30) )
        
    return bounds


def display_callback(a):
    
    
    print('One iteration completed')
    
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
sys.traj.x[:,2] = dec[2*grid:3*grid]
sys.traj.x[:,3] = dec[3*grid:4*grid]
sys.traj.u[:,0] = dec[4*grid:5*grid]
sys.traj.u[:,1] = dec[5*grid:6*grid]


sys.plot_trajectory('xu')
sys.animate_simulation()