#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:12:11 2021

@author: alexandregirard
"""

import numpy as np
from scipy.optimize import minimize

n = 2
m = 1
grid = 100
dt   = 0.01

mass = 1
target = 1

#dec = np.linspace(0,grid*3,grid*3)

count = 0


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
        
        #J = J + 0.5 * dt * ( (x[0,i]-target)**2 + (x[0,i+1]-target)**2 + x[1,i]**2 + x[1,i+1]**2 + u[0,i]**2 + u[0,i+1]**2 )
        
        J = J + 0.5 * dt * ( u[0,i]**2 + u[0,i+1]**2 ) 
        
    return J

def constraints(dec):
    
    x,u = dec2xu(dec)
    
    vec=np.zeros(grid*2-2)
    
    for i in range(grid-1):
        
        vec[i]      = (x[0,i+1] - x[0,i]) - 0.5 * dt * (  x[1,i+1] + x[1,i] )
        vec[i+grid-1] = (x[1,i+1] - x[1,i]) - 0.5 * dt * (  u[0,i+1]/mass + u[0,i]/mass )
        
    return vec


def compute_bounds():
    bounds = []
    
    bounds.append( (0,0.01) )
    
    for i in range(1,grid-1):
        
        bounds.append( (0,10) )
        
    bounds.append( (0.99,1) )
        
    bounds.append( (0,0.01) )
    
    for i in range(1,grid-1):
        
        bounds.append( (-100,100) )
        
    bounds.append( (0,0.01) )
        
    for i in range(grid):
        
        bounds.append( (-100,100) )
        
    return bounds

def display_callback(a):
    
    
    print('One iteration completed')
    
    return True


# Guess
dec = np.zeros(grid*(n+m))
dec[0:grid] = np.linspace(0,1,grid)
dec[grid:2*grid] = 1
dec[2*grid:3*grid] = 1

bnds = compute_bounds()

cons = {'type': 'eq', 'fun': constraints }
res = minimize( cost, dec, method='SLSQP',  bounds=bnds, constraints=cons, callback=display_callback ) #

print(res)

from pyro.dynamic  import integrator

sys = integrator.DoubleIntegrator()

sys.compute_trajectory(1,100)


dec = res.x
    
sys.traj.x[:,0] = dec[0:grid]
sys.traj.x[:,1] = dec[grid:2*grid]
sys.traj.u[:,0] = dec[2*grid:3*grid]

sys.plot_trajectory('xu')
