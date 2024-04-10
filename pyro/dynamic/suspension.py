#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:02:03 2022

@author: alex
"""

##############################################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
##############################################################################
from pyro.dynamic import system
##############################################################################


##############################################################################
        
class QuarterCarOnRoughTerrain( system.ContinuousDynamicSystem ):
    """ 

    """
    
    ############################
    def __init__(self):
        """ """
    
        # Dimensions
        self.n = 3   
        self.m = 1   
        self.p = 3
        
        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, self.n, self.m, self.p)
        
        # Labels
        self.name = 'Quarter Car on Rought Terrain'
        self.state_label = ['dy','y','x']
        self.input_label = ['f']
        self.output_label = self.state_label
        
        # Units
        self.state_units = ['[m/sec]','[m]','[m]']
        self.input_units = ['[N]']
        self.output_units = self.state_units
        
        # State working range
        self.x_ub = np.array([+10,+10,+10])
        self.x_lb = np.array([-10,-10,-10])
        
        # Model param
        self.mass  = 1
        self.b  = 1
        self.k  = 1
        self.vx = 1
        
        # Graphic output parameters 
        self.dynamic_domain  = True
        self.dynamic_range   = 10
        
        # Ground
        self.a   = np.array([ 0.5 , 0.3 , 0.7 , 0.2 , 0.2 , 0.1 ]) # amplitude
        self.w   = np.array([ 0.2 , 0.4 , 0.5 , 1.0 , 2.0 , 3.0 ]) # spatial freq
        self.phi = np.array([ 3.0 , 2.0 , 0.0 , 0.0 , 0.0 , 0.0 ]) # phase
        
        
    ###########################################################################
    # Ground
    ###########################################################################
        
    #############################
    def z(self, x ):
        """ get ground level at a given x """
        
        z = 0 
        
        for i in range(self.a.size):
            z  =  z + self.a[i] * np.sin( self.w[i] * ( x - self.phi[i] ))
            
        return z
        
        
    #############################
    def dz(self, x ):
        """ get ground slope at a given x """
        
        dz = 0 
        
        for i in range(self.a.size):
            dz  =  dz + self.a[i] * self.w[i] * np.cos( self.w[i] * ( x - self.phi[i] ))
            
        return dz
    

    #############################
    def f(self, x = np.zeros(3) , u = np.zeros(1) , t = 0 ):
        """ 
        Continuous time foward dynamics evaluation
        
        dx = f(x,u,t)
        
        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1
        
        OUPUTS
        dx : state derivative vectror n x 1
        
        """
        
        dx = np.zeros(self.n) # State derivative vector
        
        z  = self.z( x[2] )
        dz = self.dz( x[2] )

        dx[0] = 1./self.mass * ( u[0] - self.k * ( x[1] - z ) - self.b * ( x[0] - dz ))
        dx[1] = x[0]
        dx[2] = self.vx
        
        return dx
    
    
    ###########################################################################
    # For graphical output
    ###########################################################################
    
    
    #############################
    def xut2q( self, x , u , t ):
        """ compute config q """
        
        q   = np.array([x[2],x[1]])
        
        return q
    
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.dynamic_range
        
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
        
        lines_pts   = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        ###########################
        # Ground line
        ###########################
        
        n = 100
            
        pts = np.zeros((n,3))
        
        x = np.linspace(q[0]-self.dynamic_range*1.3,q[0]+self.dynamic_range*1.3,n)
        
        z  = np.zeros(n)
        
        for i in range(n):
            z[i]  = self.z( x[i] )
        
        pts[:,0] = x
        pts[:,1] = z
        
        lines_pts.append( pts )
        lines_style.append('-')
        lines_color.append('r')
        
        ###########################
        # mass
        ###########################
        
        pts      = np.zeros(( 5 , 3 ))
        
        l = 0.5
        h = 3.0
        
        xyz = np.array([ q[0] , q[1] + h , 0 ])
        
        pts[0,:] =  np.array([-l,+l,0]) + xyz
        pts[1,:] =  np.array([+l*2,+l,0]) + xyz
        pts[2,:] =  np.array([+l*2,-l,0]) + xyz
        pts[3,:] =  np.array([-l,-l,0]) + xyz
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        ###########################
        # spring
        ###########################
        
        # spring
        pts      = np.zeros(( 15 , 3 ))
        
        d = 0.3
        x = q[0]
        z = self.z( x )
        h =  3.0 + q[1] - 0.5 - z
        
        pts[0,:]  = np.array([x+0,z+h*0.00,0])
        pts[1,:]  = np.array([x+0,z+h*0.20,0])
        pts[2,:]  = np.array([x+d,z+h*0.25,0])
        pts[3,:]  = np.array([x-d,z+h*0.30,0])
        pts[4,:]  = np.array([x+d,z+h*0.35,0])
        pts[5,:]  = np.array([x-d,z+h*0.40,0])
        pts[6,:]  = np.array([x+d,z+h*0.45,0])
        pts[7,:]  = np.array([x-d,z+h*0.50,0])
        pts[8,:]  = np.array([x+d,z+h*0.55,0])
        pts[9,:]  = np.array([x-d,z+h*0.60,0])
        pts[10,:] = np.array([x+d,z+h*0.65,0])
        pts[11,:] = np.array([x-d,z+h*0.70,0])
        pts[12,:] = np.array([x+d,z+h*0.75,0])
        pts[13,:] = np.array([x,z+h*0.80,0])
        pts[14,:] = np.array([x,z+h*1.00,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')

                
        return lines_pts, lines_style , lines_color
    
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        plots the force vector
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        # force arrow
        pts      = np.zeros(( 5 , 3 ))
        
        f  = u[0] # force amplitude
        xf = x[2] + 0.5 # base of force x coordinate
        yf = x[1] + 3.0 - 0.5 -f # base of force y coordinate
        
        
        d = 0.2
        
        pts[0,:] =  np.array([ xf       , yf           , 0 ])
        pts[1,:] =  np.array([ xf       , yf + f       , 0 ])
        pts[2,:] =  np.array([ xf + d*f , yf + f - d*f , 0 ])
        pts[3,:] =  np.array([ xf       , yf + f       , 0 ])
        pts[4,:] =  np.array([ xf - d*f , yf + f - d*f , 0 ])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'r')
                
        return lines_pts , lines_style , lines_color
    
    
    #############################
    def plot_ground(self, x_min = 0 , x_max = 10 , n = 200 ):
        """ Plot the ground """
        
        fig , ax = plt.subplots(2, sharex=True, figsize=(6,4), dpi=200, frameon=True)
        
        fig.canvas.manager.set_window_title('Ground')
        
        x = np.linspace(x_min,x_max,n)
        
        z  = np.zeros(n)
        dz = np.zeros(n)
        
        for i in range(n):
            z[i]  = self.z( x[i] )
            dz[i] = self.dz( x[i] )
        
        
        ax[0].plot( x , z , 'b')
        ax[0].set_ylabel('z')
        ax[0].axis('equal')
        ax[0].grid(True)
        ax[0].tick_params( labelsize = 8 )
        
        ax[1].plot( x , dz , 'r')
        ax[1].set_ylabel('dz')
        ax[1].axis('equal')
        ax[1].grid(True)
        ax[1].tick_params( labelsize = 8 )
        
        fig.show()
    
    
    
    

'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    def t2u(t):
        return np.array([t])
    
    sys = QuarterCarOnRoughTerrain()
    
    sys.plot_ground()
    
    sys.show([0,0])
    
    sys.mass = 1.0
    sys.b    = 2.0
    sys.k    = 10.0
    
    sys.vx = 2.0
    
    sys.t2u = t2u
    
    sys.compute_trajectory()
    
    sys.animate_simulation()