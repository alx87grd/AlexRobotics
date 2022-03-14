#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:46:42 2022

@author: alex
"""

##############################################################################
import numpy as np
##############################################################################
from pyro.dynamic import system
##############################################################################

import matplotlib
import matplotlib.pyplot as plt



##############################################################################
# 1 DoF Car Model
##############################################################################
        
class LongitudinalFrontWheelDriveCarWithWheelSlipInput( system.ContinuousDynamicSystem ):
    """ 
    Equations of Motion
    -------------------------
    TBD
    """
    
    ############################
    def __init__(self):
        """ """
    
        # Dimensions
        self.n = 2   
        self.m = 1   
        self.p = 2
        
        # initialize standard params
        super().__init__(self.n, self.m, self.p)
        
        # Labels
        self.name = 'Front Wheel Drive Car'
        self.state_label = ['x','dx']
        self.input_label = ['slip']
        self.output_label = ['x','dx']
        
        # Units
        self.state_units = ['[m]','[m/sec]']
        self.input_units = ['[]']
        self.output_units = ['[m]','[m/sec]']
        
        # State working range
        self.x_ub = np.array([+50,+30,])
        self.x_lb = np.array([ 0,-10])
        
        # Input working range
        self.u_ub = np.array([  0.3])
        self.u_lb = np.array([ -0.3])
        
        # Model param
        self.lenght  = 2          # distance between front wheel and back wheel [m]
        self.xc      = 1          # distance from back wheel to c.g.  [m]
        self.yc      = 0.5        # height from ground to c.g.  [m]
        self.mass    = 1500       # total car mass [kg]
        self.gravity = 9.81       # gravity constant [N/kg]
        self.rho     = 1.225      # air density [kg/m3]
        self.cdA     = 0.3 * 2    # drag coef time area [m2]
        
        # Ground traction curve parameters
        self.mu_max   = 0.5
        self.mu_slope = 70.
        
        
        # Graphic output parameters 
        self.dynamic_domain  = False
        self.dynamic_range   = self.lenght * 2
        
        # Animation output graphical parameters
        self.linestyle = '-'
        self.obs_dist =  self.x_ub[0] + self.lenght * 2 # using the upper bound on x range
        self.obs_size = 2
        
        
    #############################
    def compute_ratios(self):
        """ Shorcut function for comuting usefull length ratios """
        
        ry = self.yc / self.lenght
        rr = self.xc / self.lenght # ratio of space rear of the c.g.
        rf = 1 - rr                # ratio of space in front of c.g.
        
        return ry, rr, rf
    
    #############################
    def slip2force(self, slip ):
        """ Shorcut function for comuting usefull length ratios """
        
        # Sigmoid
        mu = self.mu_max * ( 2 / ( 1 + np.exp( -self.mu_slope * slip ) ) - 1 )
        
        return mu
    
    
    #############################
    def plot_slip2force(self, slip_min = -0.2, slip_max = 0.2 ):
        """ Shorcut function for comuting usefull length ratios """
        
        slips = np.arange( slip_min, slip_max, 0.001 )
        mus   = self.slip2force( slips )
        
        fig = plt.figure(figsize=(4, 2), dpi=300, frameon=True)
        fig.canvas.manager.set_window_title('Ground traction curve')
        ax  = fig.add_subplot(1, 1, 1)
        
        ax.plot( slips , mus , 'b')
        ax.set_ylabel('mu = |Fx/Fz|', fontsize=5)
        ax.set_xlabel('Slip ratio', fontsize=5 )
        ax.tick_params( labelsize = 5 )
        ax.grid(True)
        
        fig.tight_layout()
        fig.canvas.draw()
        
        plt.show()
    
    
        
    #############################
    def f(self, x , u , t = 0 ):
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
        
        ###################
        
        slip = u
        v    = x[1]
        
        # compute ratio of horizontal/vertical force
        mu = self.slip2force( slip ) 
        
        # constant params local vairables
        ry, rr, rf = self.compute_ratios() 
        m    = self.mass 
        g    = self.gravity
        rcda = self.rho * self.cdA
        
        # Drag froce
        fd = 0.5 * rcda * v * np.abs( v ) # drag froce with the right sign
        
        # Acceleration (equation considering weight transfer)
        a  = (mu * m * g * rr - fd )/( m * (1 + mu * ry ))
        
        ###################
        
        dx[0]  = v # velocity
        dx[1]  = a # acc
        
        ###################
        # Normal force check
        fn_front = m * g * rr - m * a * ry
        fn_rear  = m * g * rf + m * a * ry
        if (fn_front<0) :
            print('Normal force on front wheel is negative: fn = ', fn_front)
        if (fn_rear<0) : 
            print('Normal force on rear wheel is negative: fn = ', fn_rear)
        ###################
        
        return dx
    
    
    ###########################################################################
    # For graphical output
    ###########################################################################
    
    
    #############################
    def xut2q( self, x , u , t ):
        """ compute config q """
        
        q   = np.append(  x , u[0] ) # steering angle is part of the config
        
        return q
    
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.dynamic_range
        
        x = q[0]
        y = 0
        z = 0
        
        if self.dynamic_domain:
        
            domain  = [ ( -l + x , l + x ) ,
                        ( -l + y , l + y ) ,
                        ( -l + z , l + z ) ]#  
        else:
            
            domain  = [ ( 0 , self.obs_dist + self.obs_size * 2 ) ,
                        ( 0 , 1 ) ,
                        ( 0 , self.obs_dist + self.obs_size * 2 ) ]#
            
                
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
        
        # Variables
        
        travel   = q[0]
        slipping = (np.abs( q[2] ) > 0.03 ) # bool
        
        
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        
        
        ###########################
        # bottom line
        ###########################
        
        pts = np.zeros((2,3))
        
        pts[0,:] = [ -1000  , 0 , 0 ]
        pts[1,:] = [  1000  , 0 , 0 ]
        
        lines_pts.append( pts )
        
        ###########################
        # obstacle
        ###########################
        
        pts = np.zeros((5,3))
        
        d = self.obs_size
        
        pts[0,:] = [ 0  , 0 , 0 ]
        pts[1,:] = [ d  , 0 , 0 ]
        pts[2,:] = [ d  , d , 0 ]
        pts[3,:] = [ 0  , d , 0 ]
        pts[4,:] = [ 0  , 0 , 0 ]
        
        pts[:,0] = pts[:,0] + self.obs_dist

        
        lines_pts.append( pts )
        
        ###########################
        # Vehicule
        ###########################
        
        pts = np.zeros((13,3))
        
        r = 0.3
        x1 = 1
        y1 = 1
        y2 = 1.5
        y3 = 1.4
        x2 = 1
        x3 = 1
        y3 = 0.6
        
        l = self.lenght
        
        pts[0,:]  = [ 0  , 0 , 0 ]
        pts[1,:]  = [ -x1  , 0 , 0 ]
        pts[2,:]  = [ -x1  , y1 , 0 ]
        pts[3,:]  = [ 0  , y2 , 0 ]
        pts[4,:]  = [ l , y2 , 0 ]
        pts[5,:]  = [ l - x2 , y2 , 0 ]
        pts[6,:]  = [ l - x2  , y1 , 0 ]
        pts[7,:]  = [ l  , y1 , 0 ]
        pts[8,:]  = [ l  , y2 , 0 ]
        pts[9,:]  = [ l  , y1 , 0 ]
        pts[10,:] = [ l+x3  , y3 , 0 ]
        pts[11,:] = [ l+x3  , 0 , 0 ]
        pts[12,:] = [ 0  , 0 , 0 ]


        pts[:,0] = pts[:,0] + travel  # translate horizontally the car postion
        pts[:,1] = pts[:,1] + r       # translate vertically the wheel radius
        
        lines_pts.append( pts )
        
        ###########################
        # Wheels
        ###########################
        
        if slipping:
            r = r*1.2
        
        angles = np.arange(0,6.4,0.1)
        n      = angles.size
        
        pts = np.zeros((n,3))
        
        for i in range(n):
            a = angles[i]
            pts[i,:] = [ r * np.cos(a) , r * np.sin(a) , 0 ]

        pts[:,0] = pts[:,0] + travel
        pts[:,1] = pts[:,1] + r
        
        lines_pts.append( pts )
        
        pts = pts.copy()
        pts[:,0] = pts[:,0] + l
        
        lines_pts.append( pts )
                
        return lines_pts
    
    

##############################################################################
# 2 DoF Car Model
##############################################################################
        
class LongitudinalFrontWheelDriveCarWithTorqueInput( LongitudinalFrontWheelDriveCarWithWheelSlipInput ):
    """ 
    Equations of Motion
    -------------------------
    TBD
    """
    
    ############################
    def __init__(self):
        """ """
        
        # initialize standard params
        super().__init__()
        
        # Dimensions
        self.n = 4   
        self.m = 1   
        self.p = 1
        
        # initialize standard vectors with new dimensions
        system.ContinuousDynamicSystem.__init__(self,self.n,self.m,self.p)

        # Labels
        self.name = 'Front Wheel Drive Car'
        self.state_label = ['x','dx', 'w','theta']
        self.input_label = ['torque']
        self.output_label = ['slip']
        
        # Units
        self.state_units = ['[m]','[m/sec]','[rad/sec]','[rad]']
        self.input_units = ['[Nm]']
        self.output_units = ['']
        
        # State working range
        self.x_ub = np.array([+50,+30,1000, 10000])
        self.x_lb = np.array([ 0,-10,-100,-10000])
        
        # Input working range
        self.u_ub = np.array([  200])
        self.u_lb = np.array([ -200])
        
        # Additionnal Model param
        self.wheel_radius  = 0.3 # [m]
        self.wheel_inertia = 1.5 # [kg m2]
        
        self.x0 = np.array([0,0.01,0,0])
        
        self.linestyle = '-'
        self.dynamic_domain = True
        
        
    #############################
    def f(self, x , u , t = 0 ):
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
        
        ###################
        
        torque = u
        v      = x[1]
        w      = x[2]
        
        # constant params local vairables
        ry, rr, rf = self.compute_ratios() 
        m    = self.mass 
        g    = self.gravity
        rcda = self.rho * self.cdA
        r    = self.wheel_radius
        j    = self.wheel_inertia
        
        #slip computation
        slip = np.clip( ( r * w - v ) / (np.abs(v) + 0.0 ) , -0.5 , 0.5 )
        
        # compute ratio of horizontal/vertical force
        mu = self.slip2force( slip ) 
        
        # Drag froce
        fd = 0.5 * rcda * v * np.abs( v ) # drag froce with the right sign
        
        # Acceleration (equation considering weight transfer)
        a  = (mu * m * g * rr - fd )/( m * (1 + mu * ry ))
        
        # Wheel acceleration
        dw = (torque - r * (m * a + fd)) / j
        
        ###################
        
        dx[0]  = v   # velocity
        dx[1]  = a   # acc
        dx[2]  = dw  # angular acc. of the wheels
        dx[3]  = w   #
        
        ###################
        # Normal force check
        fn_front = m * g * rr - m * a * ry
        fn_rear  = m * g * rf + m * a * ry
        if (fn_front<0) :
            print('Normal force on front wheel is negative: fn = ', fn_front)
        if (fn_rear<0) : 
            print('Normal force on rear wheel is negative: fn = ', fn_rear)
        ###################
        
        return dx
    
    
    #############################
    def h(self, x , u , t = 0 ):
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
        
        y = np.zeros(self.p) # State derivative vector
        
        ###################
        v      = x[1]
        w      = x[2]
        
        # constant params local vairables
        r    = self.wheel_radius
        
        #slip computation
        slip = np.clip( ( r * w - v ) / (np.abs(v) + 0.0 ) , -0.5 , 0.5 )
        
        y[0] = slip
        
        return y
    
    
    ###########################################################################
    # For graphical output
    ###########################################################################
    
    
    #############################
    def xut2q( self, x , u , t ):
        """ compute config q """
        
        q   =  x
        
        return q
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        # Variables
        
        travel   = q[0]
        ang      = q[3]
        
        # constant
        
        r = self.wheel_radius
        l = self.lenght
        
        #base plot form mother class
        lines_pts = LongitudinalFrontWheelDriveCarWithWheelSlipInput.forward_kinematic_lines(self, q)
        
        
        ###########################
        # Wheels
        ###########################
        
        
        angles = np.arange(0,6.4,0.8)
        n      = angles.size
        
        pts = np.zeros((n*2,3))
        
        for i in range(n):
            a = angles[i] - ang
            pts[i*2,:]  = [ r * np.cos(a) , r * np.sin(a) , 0 ]
            pts[i*2+1,:] = [ 0 , 0 , 0 ]

        pts[:,0] = pts[:,0] + travel + l
        pts[:,1] = pts[:,1] + r
        
        
        lines_pts.append( pts )
                
        return lines_pts
    
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    sys = LongitudinalFrontWheelDriveCarWithWheelSlipInput()
    
    sys.plot_slip2force()
    
    sys.x0[1]   = 20
    sys.ubar[0] = -0.1
    
    sys.compute_trajectory( 10, 10001, 'euler')
    
    sys.plot_trajectory('xu')
    
    sys.animate_simulation()