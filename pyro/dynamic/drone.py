#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 05:43:08 2021

@author: alex
"""

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
from pyro.dynamic import system
from pyro.dynamic import mechanical
###############################################################################


##############################################################################
# 2D planar drone
##############################################################################
        
class Drone2D( mechanical.MechanicalSystem ):
    """ 
    Equations of Motion
    -------------------------
    
    """
    
    ############################
    def __init__(self):
        """ """
    
        # Dimensions
        dof       = 3   
        actuators = 2
        
        # initialize standard params
        super().__init__( dof, actuators )
        
        # Labels
        self.name = '2D drone model'
        self.state_label = ['x','y','theta','vx','vy','w']
        self.input_label = ['T1', 'T2']
        self.output_label = self.state_label
        
        # Units
        self.state_units = ['[m]','[m]','[rad]','[m/sec]','[m/sec]','[rad/sec]']
        self.input_units = ['[N]', '[N]']
        self.output_units = self.state_units
        
        # State working range
        self.x_ub = np.array([+5,+2,+1,10,10,10])
        self.x_lb = np.array([-5,-2,-1,-10,-10,-10])
        
        # Model param
        self.mass           = 1
        self.inertia        = 0.1
        self.truster_offset = 0.5
        self.gravity        = 9.81
        self.cda            = 0.1
        
        # Kinematic param
        self.width  = 1
        self.height = 0.2
        
        # Graphic output parameters 
        self.dynamic_domain  = True
        self.dynamic_range   = 10
        
        
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        H = np.zeros((3,3))
        
        H[0,0] = self.mass
        H[1,1] = self.mass
        H[2,2] = self.inertia
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        C = np.zeros((3,3))

        return C
    
    
    ###########################################################################
    def B(self, q ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.zeros((3,2))
        
        # TODO PLACE HOLDER
        B[0,0] = -np.sin( q[2] )
        B[0,1] = -np.sin( q[2] )
        B[1,0] = np.cos( q[2] )
        B[1,1] = np.cos( q[2] )
        B[2,0] = -self.truster_offset
        B[2,1] = self.truster_offset
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        g = np.zeros(3)
        
        g[1] =  self.mass * self.gravity

        return g
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d = np.zeros(3)
        
        d[0] = dq[0]*abs(dq[0]) * self.cda + dq[0] * 0.01
        d[1] = dq[1]*abs(dq[1]) * self.cda + dq[1] * 0.01
        d[2] = dq[2]*abs(dq[2]) * 0        + dq[2] * 0.01
        
        return d
    
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.width * 3
        
        x = q[0]
        y = q[1]
        z = 0
        
        if self.dynamic_domain:
        
            domain  = [ ( -l + x , l + x ) ,
                        ( -l + y , l + y ) ,
                        ( -l + z , l + z ) ]#  
        else:
            
            domain  = [ ( -l , l ) ,
                        ( -l , l ) ,
                        ( -l , l ) ]#
                
        return domain
    
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        
        ###############################
        # ground line
        ###############################
        
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-10,0,0])
        pts[1,:] = np.array([+10,0,0])
        
        lines_pts.append( pts )
        
        ###########################
        # drone body
        ###########################
        
        x = q[0]
        y = q[1]
        s = np.sin(q[2])
        c = np.cos(q[2])
        l = self.width
        h = self.height
        
        pts      = np.zeros(( 4 , 3 ))
        pts[0,:] = np.array([x+l*c-h*s,y+l*s+h*c,0])
        pts[1,:] = np.array([x+l*c,y+l*s,0])
        pts[2,:] = np.array([x-l*c,y-l*s,0])
        pts[3,:] = np.array([x-l*c-h*s,y-l*s+h*c,0])
        
        
        lines_pts.append( pts )
        
        ###########################
        # drone cg
        ###########################
        
        pts      = np.zeros(( 1 , 3 ))
        pts[0,:] = np.array([x,y,0])
        
        lines_pts.append( pts )
                
        return lines_pts
    
    
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    sys = Drone2D()
    
    sys.x0[5] = 0
    
    sys.ubar[0] = 9.81 * 0.6
    sys.ubar[1] = 9.81 * 0.7
    
    sys.plot_trajectory()
    sys.animate_simulation()