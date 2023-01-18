#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:24:05 2022

@author: alex
"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

fig , ax = plt.subplots(2, sharex=True, figsize=(6,4), dpi=200, frameon=True)

fig.canvas.manager.set_window_title('Figure Name')

n = 200

x = np.linspace(0,10,n)

a   = np.array([ 0.5 , 0.3 , 0.7 , 0.2 , 0.2 , 0.1 ])
w   = np.array([ 0.2 , 0.4 , 0.5 , 1.0 , 2.0 , 3.0 ])
phi = np.array([ 3.0 , 2.0 , 0.0 , 0.0 , 0.0 , 0.0 ])

y  = np.zeros(n)
dy = np.zeros(n)

for i in range(a.size):
    y  =  y + a[i] * np.sin( w[i] * ( x - phi[i] ))
    dy = dy + a[i] * w[i] * np.cos( w[i] * ( x - phi[i] ))


ax[0].plot( x , y , 'b')
ax[0].set_ylabel('y')
ax[0].axis('equal')
ax[0].grid(True)
ax[0].tick_params( labelsize = 8 )

ax[1].plot( x , dy , 'r')
ax[1].set_ylabel('dy')
ax[1].axis('equal')
ax[1].grid(True)
ax[1].tick_params( labelsize = 8 )

fig.show()