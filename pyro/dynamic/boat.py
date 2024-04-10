# -*- coding: utf-8 -*-

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
from pyro.dynamic   import rigidbody
from pyro.kinematic import geometry
from pyro.kinematic import drawing
from pyro.analysis  import graphical
###############################################################################

###############################################################################
        
class Boat2D( rigidbody.RigidBody2D ):    
    """
    Simple planar (3 DoF) boat model 
    
    Partialy based on
    'Low-Speed Maneuvering Models for Dynamic Positioning (3 DOFs)'
    see Section 6.7 in
    Fossen, Thor I. Handbook of marine craft hydrodynamics and motion control 
    2nd Editions. John Wiley & Sons, 2021.

    with Equation of motion (in body frame) described by:
    -------------------------------------------------------
    M(q) dv + C(q,v) v + d(q,v) = B(q) u
    dq = N(q) v
    -------------------------------------------------------
    v        :  dim = (3, 1)     : velocity variables ( foward speed, sway speed, yaw rate) in body frame
    q        :  dim = (3, 1)     : position variables ( x , y , theta ) in global frame
    u        :  dim = (2, 1)     : thruster force vector in body frame
    d(q,v)   :  dim = (3, 1)     : state-dependent dissipative forces in body frame
    M(q)     :  dim = (3, 3)     : inertia matrix in body frame
    C(q,v)   :  dim = (3, 3)     : corriolis matrix in body frame
    B(q)     :  dim = (3, 2)     : actuator matrix in body frame
    N(q)     :  dim = (3, 3)     : transformation matrix from body frame to global frame

    with the following hypothesis:
    - The boat is a planar rigid body with 3 DoF ( surge, sway and yaw)
    - The input is a 2D force vector [ F_x , F_y ] applied at a distance l_t behind the CG
    - The boat is subject to linear and quadratic damping
    - The default quadratic hydronamic forces coef. are taken from
        Fossen, Thor I. Handbook of marine craft hydrodynamics and motion control
        2nd Editions. John Wiley & Sons, 2021.
        See 6.7.1 and Fig. 6.11
        A rough fit on experimental data from a tanker
        - TODO: not fully validated
    - The c.g., c.p and body-frame origin are coincident

    """

    ############################
    def __init__(self):
        """ """

        rigidbody.RigidBody2D.__init__( self , force_inputs = 2, other_inputs = 0)

        self.input_label = ['Tx','Ty']
        self.input_units = ['[N]','[N]']

        # Dynamic properties
        self.mass     = 10000.0
        self.inertia  = 10000.0 * 1.0 ** 2
        self.l_t      = 3.0     # Distance between CG and Thrust vector
        
        # Hydrodynamic coefficients

        # linear damping
        self.damping_coef = np.array([ 2000.0, 20000.0, 10000.0 ])

        # quadratic damping
        self.Cx_max = 0.5 
        self.Cy_max = 0.6
        self.Cm_max = 0.1

        self.rho = 1000.0          # water density
        self.Alc = self.l_t * 2    # lateral area
        self.Afc = 0.25 * self.Alc  # frontal area
        self.loa = self.l_t * 2    # length over all

        # current velocity in world frame
        # TODO: not implemented
        # self.v_current = np.array([0,0,0])    

        # Graphic output parameters 
        self.width           = self.Afc
        self.height          = self.l_t * 2
        self.dynamic_domain  = True
        self.dynamic_range   = 10
        
        l   = self.height * 0.5
        w   = self.width  * 0.5
        pts      = np.zeros(( 6 , 3 ))
        pts[0,:] = np.array([-l, +w,0])
        pts[1,:] = np.array([-l, -w,0])
        pts[2,:] = np.array([+l, -w,0])
        pts[3,:] = np.array([l+w,0,0])
        pts[4,:] = np.array([+l, +w,0])
        pts[5,:] = np.array([-l, +w,0])
        
        self.drawing_body_pts = pts

        self.show_hydrodynamic_forces = False

         # State working range
        xy_range = l * 3
        self.x_ub = np.array([+xy_range,+xy_range,+np.pi,10,10,10])
        self.x_lb = np.array([-xy_range,-xy_range,-np.pi,-10,-10,-10])

        # Input working range
        self.u_ub = np.array([+10000,+1000])
        self.u_lb = np.array([-10000,-1000])

    ###########################################################################
    def B(self, q , u ):
        """ 
        Actuator Matrix 
        ------------------
        Here u is a 2D point force [ F_x , F_y ]
        applied at a point located at a distance l_t behind the CG
        hence also creating a yaw moment
        """
        
        B = np.zeros((3,2))
        
        B[0,0] = 1
        B[1,1] = 1
        B[2,1] = - self.l_t 
        
        return B
    
    ###########################################################################
    def CurrentCoef(self, alpha ):
        
        # Cl = np.sin( 2 * alpha )     # flat plate approx
        # Cd = 1 - np.cos( 2 * alpha ) # flat plate approx
        # Cm = 0.0

        # Fig. 7.6 from Fossen  (1st edition)
        # Fig. 6.11 from Fossen (2nd edition)
        Cx = - self.Cx_max * np.cos( alpha  ) * np.abs( np.cos( alpha ) ) 
        Cy = + self.Cy_max * np.sin( alpha  ) * np.abs( np.sin( alpha ) )
        Cm = + self.Cm_max * np.sin( 2 * alpha )
        
        return np.array([ Cx , Cy , Cm ])
        
    ###########################################################################
    def d(self, q , v , u ):
        """ 
        Hydrodynamic dissipative forces
        -----------------------------------

        The model is a combination of linear and quadratic damping

        """

        # linear damping 
        d_lin = v * self.damping_coef

        V2    = v[0]**2 + v[1]**2
        alpha = -np.arctan2( v[1] , v[0] )

        Cx = self.CurrentCoef( alpha )[0]
        Cy = self.CurrentCoef( alpha )[1]
        Cm = self.CurrentCoef( alpha )[2]

        # quadratic forces
        fx = -0.5 * self.rho * self.Afc * Cx * V2
        fy = -0.5 * self.rho * self.Alc * Cy * V2
        mz = -0.5 * self.rho * self.Alc * self.loa * Cm * V2

        d_quad = np.array([ fx , fy , mz ])
        
        return d_quad + d_lin 
    
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):

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
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
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
        Graphical output showing trust vectors and hydrodynamic forces
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []

        # M per Newton of force
        f2r = 1.0 / self.u_ub[0] * self.height * 0.5
        
        ###########################
        # trust force vector
        ###########################
        
        vx  = u[0] * f2r
        vy  = u[1] * f2r
        offset = -self.l_t
        
        pts_body = drawing.arrow_from_components( vx , vy , x = offset, origin = 'tip'  )    
        W_T_B    = geometry.transformation_matrix_2D( x[2], x[0] , x[1] )
        pts_W    = drawing.transform_points_2D( W_T_B , pts_body )
        
        lines_pts.append( pts_W )
        lines_style.append( '-')
        lines_color.append( 'r' )


        ###########################
        # hydro forces
        ###########################

        if self.show_hydrodynamic_forces:

            q , v = self.x2q( x )

            f = -self.d( q , v , u)

            pts_body = drawing.arrow_from_components( f[0] * f2r , f[1] * f2r )

            pts_W = drawing.transform_points_2D( W_T_B  , pts_body )

            lines_pts.append( pts_W )
            lines_style.append( '--')
            lines_color.append( 'k' )


                
        return lines_pts , lines_style , lines_color
    

    ###########################################################################
    def plot_alpha2Coefs(self, alpha_min = -3.15, alpha_max = 3.15 ):
        
        alphas = np.arange( alpha_min, alpha_max, 0.05 )
        
        n   = alphas.shape[0]
        Cxs = np.zeros((n,1))
        Cys = np.zeros((n,1))
        Cms = np.zeros((n,1))
        
        for i in range(n):
            Cxs[i] = self.CurrentCoef( alphas[i] )[0]
            Cys[i] = self.CurrentCoef( alphas[i] )[1]
            Cms[i] = self.CurrentCoef( alphas[i] )[2]
        
        fig , ax = plt.subplots(3, figsize=graphical.default_figsize,
                                dpi= graphical.default_dpi, frameon=True)
        
        fig.canvas.manager.set_window_title('Aero curve')
        
        ax[0].plot( alphas , Cxs , 'b')
        ax[0].set_ylabel('Cx', fontsize=graphical.default_fontsize)
        ax[0].set_xlabel('alpha', fontsize=graphical.default_fontsize )
        ax[0].tick_params( labelsize = graphical.default_fontsize )
        ax[0].grid(True)
        
        ax[1].plot( alphas , Cys , 'b')
        ax[1].set_ylabel('Cy', fontsize=graphical.default_fontsize)
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

    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    sys = Boat2D()

    sys.x0[0] = 0
    sys.x0[1] = 0
    sys.x0[2] = 0

    sys.x0[3] = 2.0
    sys.x0[4] = 0.0
    sys.x0[5] = 0.0
    
    sys.ubar[0] = 10000
    sys.ubar[1] = 1000

    sys.plot_alpha2Coefs()

    #sys.show_hydrodynamic_forces = True

    sys.plot_trajectory('xu')
    sys.animate_simulation()