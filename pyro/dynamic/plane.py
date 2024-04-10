#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:29:50 2023

@author: alex
"""

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
from pyro.analysis  import graphical
from pyro.dynamic   import mechanical
from pyro.kinematic import geometry
from pyro.kinematic import drawing
###############################################################################



##############################################################################
# 2D planar plane
##############################################################################
        
class Plane2D( mechanical.MechanicalSystemWithPositionInputs ):
    
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
        self.name = '2D plane model'
        self.state_label = ['x','y','theta','vx','vy','w']
        self.input_label = ['Trust', 'delta']
        self.output_label = self.state_label
        
        # Units
        self.state_units = ['[m]','[m]','[rad]','[m/sec]','[m/sec]','[rad/sec]']
        self.input_units = ['[N]', '[Rad]']
        self.output_units = self.state_units
        
        # State working range
        self.x_ub = np.array([+100,+200,+2,30,30,10])
        self.x_lb = np.array([-100,-0,-2,-30,-30,-10])
        
        self.u_ub = np.array([+10,+0.3])
        self.u_lb = np.array([ 0, -0.3])
        
        # Model param
        self.mass           = 2.0      # kg
        self.inertia        = 0.1       # kgm2
        self.gravity        = 9.8
        
        # Aero param
        self.rho            = 1.29      # air density
        
        self.S_w            = 0.2       # wing ref. area
        self.S_t            = 0.05      # tail ref. area
        self.l_w            = 0.0       # wing a.c. position with respect to c.g., negative is behind c.g.
        self.l_t            = 1.0       # tail a.c. position with respect to c.g., negative is behind c.g.
        
        # we assume same wing profile and geometry for wing and tail
        self.Cd0            = 0.02     # parasite drag
        self.AR             = 5.0      # aspect ratio
        self.e_factor       = 0.8      # oswald efficiency factor
        self.Cm0            = 0.0      # Aero moment coef. ac
        self.alpha_stall    = np.pi / 12.      
        
        
        # Graphic output parameters 
        self.length          = 2.0   
        self.l_cg            = self.length * 0.6 # distance from back of airplane to cg
        self.width           = self.length / 10.0
        self.dynamic_domain  = True
        self.dynamic_range   = self.length * 1.0
        self.static_range    = self.length * 30
        
        
    ###########################################################################
    def compute_velocity_vector(self, q , dq ):
        
        theta = q[2]
        vx    = dq[0]
        vy    = dq[1]
        
        V     = np.sqrt( vx**2 + vy**2 )  # absolute velocity
        gamma = np.arctan2( vy , vx )     # velocity vector angle
        
        alpha = theta - gamma             # angle of attack
        
        return ( V , gamma , alpha )
    
    
    ###########################################################################
    def Cl(self, alpha ):
        
        # Rough fit on
        # https://www.aerospaceweb.org/question/airfoils/q0150b.shtml
        
        Cl = np.sin( 2 * alpha ) # falt plate approx
        
        #If not stalled
        if (alpha < self.alpha_stall ) and (alpha > -self.alpha_stall ):
            
            Cl = Cl + 4 * alpha
        
        return Cl
    
    
    ###########################################################################
    def Cd(self, alpha ):
        
        Cl = self.Cl( alpha )
        
        # Body parasite drag
        Cd = self.Cd0 
        
        # Wing flat plate approx
        Cd = Cd + ( 1 - np.cos( 2 * alpha ))
        
        #If not stalled: add induced drag
        if (alpha < self.alpha_stall ) and (alpha > -self.alpha_stall ):
            
            Cd = Cd + Cl **2 / ( np.pi * self.e_factor * self.AR )
                
        
        return Cd
    
    
    ###########################################################################
    def Cm(self, alpha ):
        
        Cm = self.Cm0 
        
        return Cm
    
    
    #############################
    def plot_alpha2Cl(self, alpha_min = -3.15, alpha_max = 3.15 ):
        
        alphas = np.arange( alpha_min, alpha_max, 0.05 )
        
        n   = alphas.shape[0]
        Cls = np.zeros((n,1))
        Cds = np.zeros((n,1))
        Cms = np.zeros((n,1))
        
        for i in range(n):
            Cls[i] = self.Cl( alphas[i] )
            Cds[i] = self.Cd( alphas[i] )
            Cms[i] = self.Cm( alphas[i] )
        
        fig , ax = plt.subplots(3, figsize=graphical.default_figsize,
                                dpi= graphical.default_dpi, frameon=True)

        fig.canvas.manager.set_window_title('Aero curve')
        
        ax[0].plot( alphas , Cls , 'b')
        ax[0].set_ylabel('Cl', fontsize=graphical.default_fontsize)
        ax[0].set_xlabel('alpha', fontsize=graphical.default_fontsize )
        ax[0].tick_params( labelsize = graphical.default_fontsize )
        ax[0].grid(True)
        
        ax[1].plot( alphas , Cds , 'b')
        ax[1].set_ylabel('Cd', fontsize=graphical.default_fontsize)
        ax[1].set_xlabel('alpha', fontsize=graphical.default_fontsize )
        ax[1].tick_params( labelsize = graphical.default_fontsize )
        ax[1].grid(True)
        
        ax[2].plot( alphas , Cms , 'b')
        ax[2].set_ylabel('Cm', fontsize=graphical.default_fontsize)
        ax[2].set_xlabel('alpha', fontsize=graphical.default_fontsize )
        ax[2].tick_params( labelsize = graphical.default_fontsize )
        ax[2].grid(True)
        
        fig.tight_layout()
        fig.canvas.draw()
        
        plt.show()
    
    
    
    ###########################################################################
    def compute_aerodynamic_forces( self, V , alpha , delta ):
        
        rv2 = 0.5 * self.rho * V**2
        
        c_w = np.sqrt( self.S_w / self.AR ) 
        c_t = np.sqrt( self.S_t / self.AR ) 
        
        Cl_w = self.Cl( alpha )
        Cd_w = self.Cd( alpha )
        Cm_w = self.Cm( alpha )
        
        L_w = rv2 * self.S_w * Cl_w
        D_w = rv2 * self.S_w * Cd_w
        M_w = rv2 * self.S_w * c_w * Cm_w
        
        Cl_t = self.Cl( alpha + delta )
        Cd_t = self.Cd( alpha + delta )
        Cm_t = self.Cm( alpha + delta )
        
        L_t = rv2 * self.S_t * Cl_t
        D_t = rv2 * self.S_t * Cd_t
        M_t = rv2 * self.S_t * c_t * Cm_t
        
        return ( L_w , D_w , M_w , L_t , D_t , M_t )
        
        
        
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
        
        V , gamma , alpha = self.compute_velocity_vector( q , dq )
        
        delta = u[1]
        
        L_w, D_w, M_w, L_t, D_t, M_t = self.compute_aerodynamic_forces( V , alpha, delta )
        
        ##########################################################
        # Total aero forces vector at c.g. in wind-aligned basis
        ##########################################################
        
        s = np.sin( alpha )
        c = np.cos( alpha )
        
        L = L_w + L_t
        D = D_w + D_t
        M = M_w + M_t - self.l_w * ( L_w * c + D_w * s ) - self.l_t * ( L_t * c + D_t * s )
        
        ##########################################################
        # Transformation of aero forces in global inertial basis
        ##########################################################
        
        d_wind = np.array([ -D , L , M ]) 
        
        s = np.sin( gamma )
        c = np.cos( gamma )
        
        R = np.array([ [ c   , -s ,  0 ] , 
                       [ s   ,  c ,  0 ] ,
                       [ 0   ,  0 ,  1 ]   ])
        
        d = - R @ d_wind # aero forces in global inertial basis
        
        return d
    
    ###########################################################################
    def B(self, q , u ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.zeros((3,1))
        
        theta = q[2]
        
        # TODO PLACE HOLDER
        B[0,0] = np.cos( theta )
        B[1,0] = np.sin( theta )
        
        return B
    
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        
        x = q[0] 
        y = q[1] 
        z = 0
        
        if self.dynamic_domain:
            
            l = self.dynamic_range
        
            domain  = [ ( -l + x , l + x ) ,
                        ( -l + y , l + y ) ,
                        ( -l + z , l + z ) ]
        else:
            
            l = self.static_range
            
            domain  = [ ( -l * 0.01 , l ) ,
                        ( -l * 0.01 , l ) ,
                        ( -l * 0.01 , l ) ]#
            
                
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
        # Dimensions
        ###########################
        
        w = self.width  # body width
        l = self.length # body lenght
        
        ###########################
        # Body
        ###########################
        
        pts      = np.zeros(( 5 , 3 ))
        
        x     = q[0]
        y     = q[1]
        theta = q[2]
        
        world_T_body = geometry.transformation_matrix_2D( theta , x , y )
        #body_T_wind  = transformation_matrix_2D_from_base_angle( -alpha , 0 , 0 )
        body_T_drawing = geometry.transformation_matrix_2D( 0 , -self.l_cg , -w/2 )
        
        body_pts_local = np.array([ [ 0   ,  0   ,  1 ] , 
                                    [ l   ,  0   ,  1 ] ,
                                    [ l-w ,  w   ,  1 ] ,
                                    [ 2*w ,  w   ,  1 ] ,
                                    [ w   ,  3*w ,  1 ] ,
                                    [ 0   ,  3*w ,  1 ] ,
                                    [ 0   ,  0   ,  1 ] ])

        
        body_pts_global = drawing.transform_points_2D( world_T_body @  body_T_drawing , body_pts_local )
        
        lines_pts.append( body_pts_global )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        if self.dynamic_domain :
            
            cg = np.array([ [ x  ,  y   ,  1 ] ])
            
            lines_pts.append( cg )
            lines_style.append( 'o')
            lines_color.append( 'k')
        
        ###########################
        # Wings
        ###########################
        
        pts      = np.zeros(( 2 , 3 ))
        
        c_w = np.sqrt( self.S_w / self.AR )

        
        wings_pts_body = np.array([ [ -self.l_w + c_w   ,  0  ,  1 ] , 
                                    [ -self.l_w - c_w   ,  0  ,  1 ] ])
        
        
        wings_pts_world = drawing.transform_points_2D( world_T_body , wings_pts_body )
        
        lines_pts.append( wings_pts_world )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        ###########################
        # bottom line
        ###########################
        
        pts = np.zeros((2,3))
        
        pts[0,0] = -10000
        pts[1,0] = 10000
        pts[0,1] = 0
        pts[1,1] = 0
        
        lines_pts.append( pts )
        lines_style.append('--')
        lines_color.append('k')
        
            
        return lines_pts , lines_style , lines_color
    
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        plots the force vector
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        #w = self.width
        
        q, dq = self.x2q(x)
        
        x     = q[0]
        y     = q[1]
        theta = q[2]
        
        V , gamma , alpha = self.compute_velocity_vector( q , dq )
        
        
        world_T_body = geometry.transformation_matrix_2D( theta , x , y )
        body_T_wind  = geometry.transformation_matrix_2D( -alpha , 0 , 0 )
        
        delta = u[1]
        
        ###########################
        # Trust vector
        ###########################
        
        # Max trust --> arrow is long as airplane
        f_scale = self.length / (self.u_ub[0] - self.u_lb[0])
        
        trust_vector_lenght = u[0] * f_scale
        
        #pts  = arrow_from_tip_angle( trust_vector_lenght , theta , bx , by )
        
        trust_arrow_body = drawing.arrow_from_length_angle( trust_vector_lenght, 0, -self.l_cg, 0 , origin = 'tip')
        
        trust_arrow_world = drawing.transform_points_2D( world_T_body , trust_arrow_body )
        
        lines_pts.append( trust_arrow_world )
        lines_style.append( '-')
        lines_color.append( 'r')
        
        ###########################
        # Control surface
        ###########################
        
        c_t = np.sqrt( self.S_t / self.AR )
        
        # NOT TO scale to better see the elevator
        tail_pts_tail = np.array([ [  1.0 * c_t   ,  0  ,  1 ] , 
                                   [ -1.0 * c_t   ,  0  ,  1 ] ]) 
        
        body_T_tail = geometry.transformation_matrix_2D( delta , - self.l_t , 0 )
        
        tail_pts_global = drawing.transform_points_2D( world_T_body @ body_T_tail , tail_pts_tail )
        
        lines_pts.append( tail_pts_global )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        # ###########################
        # # Velocity vector
        # ###########################
        
        v_length = V * self.length / self.x_ub[3]
        if v_length > self.length: v_length = self.length
        
        v_pts = drawing.arrow_from_length_angle( v_length , 0 )
        
        v_world = drawing.transform_points_2D( world_T_body @ body_T_wind , v_pts )
        
        lines_pts.append( v_world  )
        lines_style.append('-')
        lines_color.append('k')
        
        # ###########################
        # # Aero forces
        # ###########################
        
        L_w, D_w, M_w, L_t, D_t, M_t = self.compute_aerodynamic_forces( V , alpha , delta )
        
        L_w_pts = drawing.arrow_from_components(               0, L_w * f_scale )
        D_w_pts = drawing.arrow_from_components(  -D_w * f_scale,             0 )
        L_t_pts = drawing.arrow_from_components(               0, L_t * f_scale )
        D_t_pts = drawing.arrow_from_components(  -D_t * f_scale,             0 )
        
        body_T_acw = geometry.transformation_matrix_2D( 0 , -self.l_w , 0  )
        body_T_act = geometry.transformation_matrix_2D( 0 , -self.l_t , 0  )
        
        L_w_pts_global = drawing.transform_points_2D( world_T_body @  body_T_acw @ body_T_wind , L_w_pts )
        D_w_pts_global = drawing.transform_points_2D( world_T_body @  body_T_acw @ body_T_wind , D_w_pts )
        
        lines_pts.append( L_w_pts_global )
        lines_style.append('-')
        lines_color.append('b')
        
        lines_pts.append( D_w_pts_global )
        lines_style.append('-')
        lines_color.append( 'r' )
        
        L_t_pts_global = drawing.transform_points_2D( world_T_body @  body_T_act @ body_T_wind , L_t_pts )
        D_t_pts_global = drawing.transform_points_2D( world_T_body @  body_T_act @ body_T_wind , D_t_pts )
        
        
        lines_pts.append( L_t_pts_global )
        lines_style.append('-')
        lines_color.append('b')
        
        lines_pts.append( D_t_pts_global )
        lines_style.append('-')
        lines_color.append('r')
        
                
        return lines_pts , lines_style , lines_color



    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    
    if True:
    
        sys = Plane2D()
        
        #sys.plot_alpha2Cl()
        
        sys.x0   = np.array([0,0,0.2,15,0,0])
        
        
        
        def t2u(t):
            
            u = np.array([ 2 * t , -0.12 * t ])
            
            return u
            
        sys.t2u = t2u
        
        #sys.gravity = 0
        
        sys.compute_trajectory( 2 , 1001 , 'euler' )
        sys.plot_trajectory('x')
        
        # sys.dynamic_domain = False
        sys.animate_simulation( time_factor_video=0.5 )
