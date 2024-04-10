#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:11:31 2023

@author: alex
"""
###############################################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
###############################################################################
from pyro.dynamic import system
from pyro.dynamic import mechanical
from pyro.dynamic import manipulator
##############################################################################

##############################################################################
# Full mountain car
##############################################################################
        
class MountainCar( manipulator.Manipulator ):
    """ 
    Equations of Motion
    -------------------------
    TBD
    """
    
    ############################
    def __init__(self):
        """ """
    
        # Dimensions
        dof = 1
        m   = 1
        e   = 2
               
        # initialize standard params
        manipulator.Manipulator.__init__( self, dof , m , e)
        
        # Labels
        self.name = 'Mountain Car'
        self.state_label = ['x','dx']
        self.input_label = ['throttle']
        self.output_label = ['x','dx']
        
        # Units
        self.state_units = ['[m]','[m/sec]']
        self.input_units = ['[N]']
        self.output_units = ['[m]','[m/sec]']
        
        # State working range
        self.x_ub = np.array([+0.2,+0.5])
        self.x_lb = np.array([-1.7,-0.5])
        
        # Input working range
        self.u_ub = np.array([  1.0])
        self.u_lb = np.array([ -1.0])
        
        # Model param
        self.mass    = 1.0          # total car mass [kg]
        self.gravity = 1.0          # gravity constant [N/kg]
        
        # Relief curve
        self.a   = 0.5
        self.w   = np.pi
        
        # Graphic output parameters 
        self.width           = 0.03
        self.dynamic_domain  = False
        self.dynamic_range   = self.width * 50
        
        
    ###########################################################################
    # Ground 
    ###########################################################################
        
    #############################################
    def z(self, x ):

        z = self.a * np.cos( self.w * x )

        return z
    
    #############################################
    def dz_dx(self, x ):

        dz = - self.a * self.w * np.sin( self.w * x )

        return dz
    
    #############################################
    def d2z_dx2(self, x ):

        d2z = - self.a * self.w ** 2 * np.cos( self.w * x )

        return d2z
    
    ###########################################################################
    # Kinematic
    ###########################################################################
    
    ##############################
    def forward_kinematic_effector(self, q ):
        
        x = q[0]
        y = self.z( x )
        
        r = np.array([x,y])
        
        return r
    
    
    ##############################
    def J(self, q ):
        
        J = np.zeros( ( self.e  , self.dof ) ) # Place holder
        
        J[0] = 1
        J[1] = self.dz_dx( q[0] )
        
        return J
    
    ###########################################################################
    # Dynamic 
    ###########################################################################
    
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        z   = self.z(       q[0] )
        dz  = self.dz_dx(   q[0] )
        ddz = self.d2z_dx2( q[0] )
        
        H = np.zeros((1,1))
        
        H[0,0] = self.mass * ( 1 + dz ** 2 )
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        z   = self.z(       q[0] )
        dz  = self.dz_dx(   q[0] )
        ddz = self.d2z_dx2( q[0] )
        
        C = np.zeros((1,1))
        
        C[0,0] = self.mass * dz * ddz * dq[0]

        return C
    
    
    ###########################################################################
    def B(self, q ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        z   = self.z(       q[0] )
        dz  = self.dz_dx(   q[0] )
        ddz = self.d2z_dx2( q[0] )
        
        B = np.diag( np.ones( self.dof ) ) #  identity matrix
        
        B[0,0] = np.sqrt( 1 + dz ** 2 )
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        z   = self.z(       q[0] )
        dz  = self.dz_dx(   q[0] )
        ddz = self.d2z_dx2( q[0] )
        
        g = np.zeros(1)
        
        g[0] = self.mass * self.gravity * dz

        return g
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d = np.zeros(1)
        
        return d
    
    ###########################################################################
    # Graphical
    ###########################################################################
    
    #############################
    def plot_ground(self, x_min = -1.7 , x_max = 0.2 , n = 200 ):
        """ Plot the ground """
        
        fig , ax = plt.subplots(1, sharex=True, figsize=(6,4), dpi=200, frameon=True)
        
        fig.canvas.manager.set_window_title('Ground')
        
        x = np.linspace(x_min,x_max,n)
        
        z  = np.zeros(n)
        dz = np.zeros(n)
        
        for i in range(n):
            z[i]  = self.z( x[i] )
            dz[i] = self.dz_dx( x[i] )
        
        
        ax.plot( x , z , 'b')
        ax.set_ylabel('z')
        ax.axis('equal')
        ax.grid(True)
        ax.tick_params( labelsize = 8 )
        
        fig.show()
        
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.dynamic_range
        
        x = q[0]
        y = self.z( q[0] )
        z = 0
        
        if self.dynamic_domain:
        
            domain  = [ ( -l + x , l + x ) ,
                        ( -l + y , l + y ) ,
                        ( -l + z , l + z ) ]#  
        else:
            
            domain  = [ ( self.x_lb[0] , self.x_ub[0] ) ,
                        ( -self.a, self.a * 1.5 ),
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
        
        lines_pts   = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        ###########################
        # Ground line
        ###########################
        
        n = 100
            
        pts = np.zeros((n,3))
        
        r = 2 * np.pi / self.w # plot 2 period
        
        x = np.linspace(q[0]-r,q[0]+r,n)
        
        z  = np.zeros(n)
        
        for i in range(n):
            z[i]  = self.z( x[i] )
        
        pts[:,0] = x
        pts[:,1] = z
        
        lines_pts.append( pts )
        lines_style.append('-')
        lines_color.append('k')
        
        ###########################
        # mass
        ###########################
        
        pts      = np.zeros(( 5 , 3 ))
        
        l = self.width
        
        xyz = np.array([ q[0] , self.z( q[0] ) , 0 ])
        
        pts[0,:] =  np.array([-l,+l,0]) + xyz
        pts[1,:] =  np.array([+l,+l,0]) + xyz
        pts[2,:] =  np.array([+l,-l,0]) + xyz
        pts[3,:] =  np.array([-l,-l,0]) + xyz
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        return lines_pts, lines_style , lines_color
    
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        show propoulsion force vectors
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        ###########################
        # drone trust force vectors
        ###########################
        
        xcg   = x[0]
        ycg   = self.z( xcg )
        theta = np.arctan( self.dz_dx( xcg ) ) 
        s = np.sin( theta )
        c = np.cos( theta )
        l = self.width * u[0] * 5.0 * ( self.u_ub[0] - self.u_lb[0] ) 
        h = l * 0.25
        
        pts      = np.zeros(( 5 , 3 ))
        pts[0,:] = np.array([xcg,ycg,0])
        pts[1,:] = pts[0,:] + np.array([l*c,l*s,0])
        pts[2,:] = pts[1,:] + np.array([-h*c,-h*s,0]) + np.array([-h*s,+h*c,0])
        pts[3,:] = pts[1,:] 
        pts[4,:] = pts[1,:] + np.array([-h*c,-h*s,0]) - np.array([-h*s,+h*c,0])
        
        lines_pts.append( pts )
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

    
    sys = MountainCar()
    
    #sys.plot_ground()
    
    sys.x0[0] = -1.0
    sys.x0[1] = -0.5
    
    sys.ubar[0] = 20.0
    
    def t2u(t):
        return np.array([np.sin(5.0*t)]) * 1.0
    
    sys.t2u = t2u
    
    sys.compute_trajectory()
    
    sys.plot_trajectory()
    
    #sys.dynamic_domain = True
    sys.animate_simulation()
        