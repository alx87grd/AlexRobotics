#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 20:08:25 2022

@author: alex
"""

import numpy as np

from pyro.dynamic import massspringdamper



m = 1
b = 0

sys = massspringdamper.FloatingSingleMass( m , b )

sys.plot_phase_plane()

sys.x0 = np.array([0,5])
sys.compute_trajectory( tf = 1 )
sys.plot_phase_plane_trajectory()
sys.animate_simulation()


m = 1
b = 0.1

sys = massspringdamper.FloatingSingleMass( m , b )

sys.plot_phase_plane()



m = 1
b = 2

sys = massspringdamper.FloatingSingleMass( m , b )

sys.plot_phase_plane()

sys.x0 = np.array([0,5])
sys.compute_trajectory( tf = 1 )
sys.plot_phase_plane_trajectory()
sys.animate_simulation()


# Pendulum

from pyro.dynamic import pendulum

sys = pendulum.SinglePendulum()

sys.plot_phase_plane()


sys.x0 = np.array([-6.28,5])
sys.compute_trajectory( tf = 3 )
sys.plot_phase_plane_trajectory()
sys.animate_simulation()

sys.x0 = np.array([0,1])
sys.compute_trajectory( tf = 5 )
sys.plot_phase_plane_trajectory()
sys.animate_simulation()


sys.d1 = 1

sys.plot_phase_plane()

sys.x0 = np.array([0,5])
sys.compute_trajectory( tf = 10 )
sys.plot_phase_plane_trajectory()
sys.animate_simulation()



# Custom

from pyro.dynamic import system


class CustomSys( system.ContinuousDynamicSystem ):
    
    ############################
    def __init__(self):
        
        # initialize standard 2 state (n=2) dynamic system
        system.ContinuousDynamicSystem.__init__( self, n = 2 )
        
        # Name and labels
        self.name = 'My custom system'
        self.state_label = [ 'State 1' , 'State 2' ]
        self.state_units = [ '[]', '[]']

    
    #############################
    def f( self , x , u , t ):
        """ 
        Continuous time foward dynamics evaluation dx = f(x,u,t)
        
        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1
        
        OUPUTS
        dx : state derivative vector  n x 1
        
        """
        
        dx = np.zeros(self.n) # State derivative vector
        
        # Your Dynamic Equation bellow:
        dx[0] = -x[0]
        dx[1] = -x[1]
        
        return dx



sys = CustomSys()

sys.plot_phase_plane()


# Closed-loop

m = 1
b = 0

sys = massspringdamper.FloatingSingleMass( m , b )

sys.p = 2
sys.C = np.diag([1,1])
sys.cost_function = None

from pyro.control import linear

ctl = linear.ProportionalController( 1 , 2)


ctl.K[0,0] = 1.0
ctl.K[0,1] = 0.5

ctl.rbar = np.array([2,0])

cl_sys = ctl + sys

cl_sys.plot_phase_plane()

cl_sys.plot_phase_plane_closed_loop()

plt.show()



cl_sys.x0 = np.array([0,5])
cl_sys.compute_trajectory( tf = 10 )
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()
