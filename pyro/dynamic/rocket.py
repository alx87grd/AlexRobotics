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
from pyro.dynamic   import mechanical
from pyro.kinematic import geometry
from pyro.kinematic import drawing
###############################################################################


##############################################################################
# 2D planar drone
##############################################################################
        
class Rocket( mechanical.MechanicalSystemWithPositionInputs ):
    """ 
    Equations of Motion
    -------------------------
    
    """
    
    ############################
    def __init__(self):
        """ """
        
        # initialize standard params
        mechanical.MechanicalSystemWithPositionInputs.__init__( self, 3 , 1 , 1 )
        
        # Labels
        self.name = '2D rocket model'
        self.state_label = ['x','y','theta','vx','vy','w']
        self.input_label = ['Trust', 'delta']
        self.output_label = self.state_label
        
        # Units
        self.state_units = ['[m]','[m]','[rad]','[m/sec]','[m/sec]','[rad/sec]']
        self.input_units = ['[N]', '[Rad]']
        self.output_units = self.state_units
        
        # State working range
        self.x_ub = np.array([+50,+100,+2,10,10,10])
        self.x_lb = np.array([-50,-0,-2,-10,-10,-10])
        
        # Model param
        self.mass           = 1000
        self.inertia        = 100
        self.ycg            = 1
        self.gravity        = 9.8
        self.cda            = 1
        
        # Kinematic param
        self.width  = 0.2
        self.height = 2.0
        
        # Graphic output parameters 
        self.dynamic_domain  = True
        self.dynamic_range   = 10
        
        # rocket drawing
        pts = np.zeros(( 10 , 3 ))
        l   = self.height
        w   = self.width
        
        pts[0,:] = np.array([ 0, -l,0])
        pts[1,:] = np.array([-w, -l,0])
        pts[2,:] = np.array([-w, +l,0])
        pts[3,:] = np.array([ 0,l+w,0])
        pts[4,:] = np.array([+w, +l,0])
        pts[5,:] = np.array([+w, -l,0])
        pts[6,:] = pts[0,:]
        pts[7,:] = pts[0,:] + np.array([-w,-w,0])
        pts[8,:] = pts[0,:] + np.array([+w,-w,0])
        pts[9,:] = pts[0,:]
        
        self.drawing_body_pts = pts
        
        
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
    def B(self, q , u ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.zeros((3,1))
        
        delta = u[1]
        
        B[0,0] = -np.sin( q[2] + delta )
        B[1,0] =  np.cos( q[2] + delta)
        B[2,0] = - self.ycg * np.sin( delta )
        
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
    def d(self, q , dq , u ):
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
        l = self.height * 3
        
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
        lines_style = []
        lines_color = []
        
        ###############################
        # ground line
        ###############################
        
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-10,0,0])
        pts[1,:] = np.array([+10,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '--')
        lines_color.append( 'k' )
        
        ###########################
        #  body
        ###########################
        
        x     = q[0]
        y     = q[1]
        theta = q[2]
        
        W_T_B    = geometry.transformation_matrix_2D( theta , x , y )
        
        pts_B    = self.drawing_body_pts
        pts_W    = drawing.transform_points_2D( W_T_B , pts_B )

        lines_pts.append( pts_W )
        lines_style.append( '-')
        lines_color.append( 'b' )
        
        ###########################
        #  C.G.
        ###########################
        
        pts      = np.zeros(( 1 , 3 ))
        pts[0,:] = np.array([x,y,0])
        
        lines_pts.append( pts )
        lines_style.append( 'o')
        lines_color.append( 'b' )
                
        return lines_pts , lines_style , lines_color
    
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        show trust vectors
        
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        ###########################
        # trust force vector
        ###########################
        
        length   = u[0] * 0.0002       # arrow length
        theta    = u[1] - 0.5 * np.pi  # arrow angle (body frame)
        y_offset = -self.height - self.width
        
        pts_body = drawing.arrow_from_length_angle( length, theta, y = y_offset )
        W_T_B    = geometry.transformation_matrix_2D( x[2], x[0] , x[1] )
        pts_W    = drawing.transform_points_2D( W_T_B , pts_body )
        
        lines_pts.append( pts_W )
        lines_style.append( '-')
        lines_color.append( 'r' )
                
        return lines_pts , lines_style , lines_color
    
    
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    sys = Rocket()
    
    sys.x0[0] = 0
    
    sys.ubar[0] = sys.mass * sys.gravity * 1.1
    sys.ubar[1] = -0.001
    
    sys.plot_trajectory()
    sys.animate_simulation()