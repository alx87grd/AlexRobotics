# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:07:39 2018

@author: nvidia
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import mechanical
###############################################################################


###############################################################################

class SinglePendulum( mechanical.MechanicalSystem ):
    """Pendulum with a point mass and inertial rod.

    Attributes
    ----------
    l1 : float
        Length of pendulum rod. Only used for display.
    lc1 : float
        Distance of point mass to pivot.
    m1 : float
        Mass value of point mass.
    I1 : float
        Moment of inertia of pendulum rod (without mass) about the pendulum pivot. The
        total inertia of the pendulum is calculated as
        ``I_pendulum = I1 + (m1 * lc1**2)``.
    gravity : float
        Constant of gravitational acceleration
    d1: float
        Damping force factor
    """

    ############################
    def __init__(self):
        """ """
               
        # initialize standard params
        super().__init__(1)
        
        # Name
        self.name = 'Single Pendulum'
        
        # params
        self.setparams()
        
            
    #############################
    def setparams(self):
        """ Set model parameters here """
        
        # kinematic
        self.l1  = 2 
        self.lc1 = 1
        
        # dynamic
        self.m1       = 1
        self.I1       = 1
        self.gravity  = 9.81
        self.d1       = 0
        
        
    ##############################
    def trig(self, q ):
        """ Compute cos and sin """
        
        c1  = np.cos( q )
        s1  = np.sin( q )

        return [c1,s1]
    
    
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        H = np.zeros((self.dof,self.dof))
        
        H[0,0] = self.m1 * self.lc1**2 + self.I1
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        C = np.zeros((self.dof,self.dof))

        return C
    
    
    ###########################################################################
    def B(self, q ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.diag( np.ones( self.dof ) ) #  identity matrix
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        g = np.zeros( self.dof ) 
        
        [c1,s1] = self.trig( q )
        
        g[0] = self.m1 * self.gravity * self.lc1 * s1

        return g
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d    = np.zeros( self.dof ) 
        
        d[0] = self.d1 * dq[0]
        
        return d
    
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = 5
        
        domain  = [ (-l,l) , (-l,l) , (-l,l) ]#  
                
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
        
        # ground line
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-10,0,0])
        pts[1,:] = np.array([+10,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '--')
        lines_color.append( 'k' )
        
        # pendulum
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([0,0,0])
        
        [c1,s1] = self.trig( q )
        
        pts[1,0] =   s1 * self.l1
        pts[1,1] = - c1 * self.l1
        
        lines_pts.append( pts )
        lines_style.append( 'o-')
        lines_color.append( 'b' )
                
        return lines_pts , lines_style , lines_color
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        show torque as a rotating arrow around the joint
        
        """
        
        lines_pts   = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        # Torque
        f         = u[0] # torque amplitude
        f_pos     = ( f > 0 )
        q         = x[0] - np.pi / 2  # rigid link angle
        max_angle = f * ( np.pi * 2 /3 / ( self.u_ub[0] ) )
        r         = self.l1 / 5.0  # radius of arc
        r1        = r/2            # length of arrows
        da        = 0.2            # angle discretization
        
        
        if abs(f) > (self.u_ub[0] * 0.05):
        
            if f_pos:
                angles = np.arange( 0, max_angle , da  ) + q 
            else:
                angles = np.arange( 0, max_angle * -1 , da  ) * -1 + q 
            n      = angles.size
            
            # Draw arc
            pts = np.zeros((n,3))
            for i , a in enumerate( angles ):
                pts[i,:] = [ r * np.cos(a) , r * np.sin(a) , 0 ]
            
            lines_pts.append( pts )
            lines_style.append( '-')
            lines_color.append( 'r' )
            
            # Draw Arrow
            c = np.cos( max_angle + q )
            s = np.sin( max_angle + q )
            
            pts = np.zeros((3,3))
            pts[1,:] = [ r * c , r * s , 0 ]
            if f_pos:
                pts[0,:] = pts[1,:] + [ -r1/2*c+r1/2*s , -r1/2*s-r1/2*c, 0 ]
                pts[2,:] = pts[1,:] + [ +r1/2*c+r1/2*s , +r1/2*s-r1/2*c, 0 ]
            else:
                pts[0,:] = pts[1,:] + [ -r1/2*c-r1/2*s , -r1/2*s+r1/2*c, 0 ]
                pts[2,:] = pts[1,:] + [ +r1/2*c-r1/2*s , +r1/2*s+r1/2*c, 0 ]
            
            lines_pts.append( pts )
            lines_style.append( '-')
            lines_color.append( 'r' )
            
        else:
            
            pts = np.zeros((3,3))
            
            lines_pts.append( pts )
            lines_style.append( '-')
            lines_color.append( 'r' )
            
            lines_pts.append( pts )
            lines_style.append( '-')
            lines_color.append( 'r' )
        
                
        return lines_pts , lines_style , lines_color
        
        
        
##############################################################################
        
class DoublePendulum( mechanical.MechanicalSystem ):
    """ 

    """
    
    ############################
    def __init__(self):
        """ """
               
        # initialize standard params
        super().__init__(2)
        
        # Name
        self.name = 'Double Pendulum'
        
        # params
        self.setparams()
                
            
    #############################
    def setparams(self):
        """ Set model parameters here """
        
        self.l1  = 1 
        self.l2  = 1
        self.lc1 = 1
        self.lc2 = 1
        
        self.m1 = 1
        self.I1 = 0
        self.m2 = 1
        self.I2 = 0
        
        self.gravity = 9.81
        
        self.d1 = 0
        self.d2 = 0
        
        
    ##############################
    def trig(self, q ):
        """ 
        Compute cos and sin usefull in other computation 
        ------------------------------------------------
        
        """
        
        c1  = np.cos( q[0] )
        s1  = np.sin( q[0] )
        c2  = np.cos( q[1] )
        s2  = np.sin( q[1] )
        c12 = np.cos( q[0] + q[1] )
        s12 = np.sin( q[0] + q[1] )
        
        return [c1,s1,c2,s2,c12,s12]
        
    
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        H = np.zeros((2,2))
        
        H[0,0] = self.m1 * self.lc1**2 + self.I1 + self.m2 * ( self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * c2 ) + self.I2
        H[1,0] = self.m2 * self.lc2**2 + self.m2 * self.l1 * self.lc2 * c2 + self.I2
        H[0,1] = H[1,0]
        H[1,1] = self.m2 * self.lc2 ** 2 + self.I2
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        h = self.m2 * self.l1 * self.lc2 * s2
        
        C = np.zeros((2,2))
        
        C[0,0] = - h  * dq[1]
        C[1,0] =   h  * dq[0]
        C[0,1] = - h * ( dq[0] + dq[1] )
        C[1,1] = 0

        return C
    
    
    ###########################################################################
    def B(self, q ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.diag( np.ones( self.dof ) ) #  identity matrix
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1 ) * self.gravity
        g2 = self.m2 * self.lc2 * self.gravity
        
        G = np.zeros(2)
        
        G[0] = - g1 * s1 - g2 * s12
        G[1] = - g2 * s12

        return G
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        D = np.zeros((2,2))
        
        D[0,0] = self.d1
        D[1,0] = 0
        D[0,1] = 0
        D[1,1] = self.d2
        
        d = np.dot( D , dq )
        
        return d
    
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = 3
        
        domain  = [ (-l,l) , (-l,l) , (-l,l) ]#  
                
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
        
        ###############################
        # ground line
        ###############################
        
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-10,0,0])
        pts[1,:] = np.array([+10,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '--' )
        lines_color.append( 'k' )
        
        ###########################
        # pendulum kinematic
        ###########################
        
        pts      = np.zeros(( 3 , 3 ))
        pts[0,:] = np.array([0,0,0])
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        pts[1,0] = self.l1 * s1
        pts[1,1] = self.l1 * c1
        
        pts[2,0] = self.l1 * s1 + self.l2 * s12
        pts[2,1] = self.l1 * c1 + self.l2 * c12
        
        lines_pts.append( pts )
        lines_style.append( 'o-' )
        lines_color.append( 'b' )
                
        return lines_pts , lines_style , lines_color
    
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        show torques as a rotating arrow around the joint
        
        """
        
        lines_pts   = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        r          = self.l1 / 5.0  # radius of arc
        r1         = r/2            # length of arrows
        da         = 0.2            # angle discretization
        
        # Torque 1
        x1         = 0
        y1         = 0
        f1         = -u[0] # torque amplitude
        f1_pos     = ( f1 > 0 )
        q1         = -x[0] +  np.pi /2     # rigid link angle
        max_angle1 = f1 * ( np.pi * 2 /3 / ( self.u_ub[0] ) )
        
        if abs(f1) > (self.u_ub[0] * 0.05):
        
            if f1_pos:
                angles = np.arange( 0, max_angle1 , da  ) + q1 
            else:
                angles = np.arange( 0, max_angle1 * -1 , da  ) * -1 + q1 
            n      = angles.size
            
            # Draw arc
            pts = np.zeros((n,3))
            for i , a in enumerate( angles ):
                pts[i,:] = [ r * np.cos(a) + x1 , r * np.sin(a) + y1 , 0 ]
            
            lines_pts.append( pts )
            lines_style.append( '-' )
            lines_color.append( 'r' )
            
            # Draw Arrow
            a = max_angle1 + q1 
            c = np.cos( a )
            s = np.sin( a )
            
            pts = np.zeros((3,3))
    
            pts[1,:] =  [ r * c + x1 , r * s + y1 , 0 ]
            if f1_pos:
                pts[0,:] = pts[1,:] + [ -r1/2*c+r1/2*s , -r1/2*s-r1/2*c, 0 ]
                pts[2,:] = pts[1,:] + [ +r1/2*c+r1/2*s , +r1/2*s-r1/2*c, 0 ]
            else:
                pts[0,:] = pts[1,:] + [ -r1/2*c-r1/2*s , -r1/2*s+r1/2*c, 0 ]
                pts[2,:] = pts[1,:] + [ +r1/2*c-r1/2*s , +r1/2*s+r1/2*c, 0 ]
            
            lines_pts.append( pts )
            lines_style.append( '-')
            lines_color.append( 'r' )
            
        else:
            
            pts = np.zeros((3,3))
            
            lines_pts.append( pts )
            lines_style.append( '-')
            lines_color.append( 'r' )
            
            lines_pts.append( pts )
            lines_style.append( '-')
            lines_color.append( 'r' )
            
        
        # Torque 2
        x2         = self.l1 * np.sin( -q1 + np.pi /2 )
        y2         = self.l1 * np.cos( -q1 + np.pi /2)
        f2         = -u[1] # torque 2 amplitude
        f2_pos     = ( f2 > 0 )
        q12        = - x[0] - x[1] +  np.pi /2  # rigid link angle
        max_angle2 = f2 * ( np.pi * 2 /3 / ( self.u_ub[1] ) )
        
        
        if abs(f2) > (self.u_ub[0] * 0.05):
        
            if f2_pos:
                angles = np.arange( 0, max_angle2 , da  ) + q12 
            else:
                angles = np.arange( 0, max_angle2 * -1 , da  ) * -1 + q12 
            n      = angles.size
            
            # Draw arc
            pts = np.zeros((n,3))
            for i , a in enumerate( angles ):
                pts[i,:] = [ r * np.cos(a) + x2 , r * np.sin(a) + y2 , 0 ]
            
            lines_pts.append( pts )
            lines_style.append( '-' )
            lines_color.append( 'r' )
            
            # Draw Arrow
            a = max_angle2 + q12 
            c = np.cos( a )
            s = np.sin( a )
            
            pts = np.zeros((3,3))
    
            pts[1,:] =  [ r * c + x2 , r * s + y2 , 0 ]
            if f2_pos:
                pts[0,:] = pts[1,:] + [ -r1/2*c+r1/2*s , -r1/2*s-r1/2*c, 0 ]
                pts[2,:] = pts[1,:] + [ +r1/2*c+r1/2*s , +r1/2*s-r1/2*c, 0 ]
            else:
                pts[0,:] = pts[1,:] + [ -r1/2*c-r1/2*s , -r1/2*s+r1/2*c, 0 ]
                pts[2,:] = pts[1,:] + [ +r1/2*c-r1/2*s , +r1/2*s+r1/2*c, 0 ]
            
            lines_pts.append( pts )
            lines_style.append( '-')
            lines_color.append( 'r' )
            
        else:
            
            pts = np.zeros((3,3))
            
            lines_pts.append( pts )
            lines_style.append( '-')
            lines_color.append( 'r' )
            
            lines_pts.append( pts )
            lines_style.append( '-')
            lines_color.append( 'r' )
            
        
                
        return lines_pts , lines_style , lines_color
    
    
    
###############################################################################

class TwoIndependentSinglePendulum( mechanical.MechanicalSystem ):
    """ Two Pendulum with a point mass and inertial rod.

    """

    ############################
    def __init__(self):
        """ """
               
        # initialize standard params
        super().__init__(2)
        
        # Name
        self.name = 'Two Independent Single Pendulum'
        
        # params
        
        # kinematic
        self.l1  = 2 
        self.lc1 = 1
        
        # dynamic
        self.m1       = 1
        self.I1       = 1
        self.gravity  = 9.81
        self.d1       = 0
        
        
    ##############################
    def trig(self, q ):
        """ Compute cos and sin """
        
        c1  = np.cos( q[0] )
        s1  = np.sin( q[0] )
        
        c2  = np.cos( q[1] )
        s2  = np.sin( q[1] )

        return [c1,s1,c2,s2]
    
    
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        H = np.zeros((self.dof,self.dof))
        
        H[0,0] = self.m1 * self.lc1**2 + self.I1
        H[1,1] = self.m1 * self.lc1**2 + self.I1
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        C = np.zeros((self.dof,self.dof))

        return C
    
    
    ###########################################################################
    def B(self, q ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.diag( np.ones( self.dof ) ) #  identity matrix
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        g = np.zeros( self.dof ) 
        
        [c1,s1,c2,s2] = self.trig( q )
        
        g[0] = self.m1 * self.gravity * self.lc1 * s1
        g[1] = self.m1 * self.gravity * self.lc1 * s2

        return g
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d    = np.zeros( self.dof ) 
        
        d[0] = self.d1 * dq[0]
        d[1] = self.d1 * dq[1]
        
        return d
    
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = 5
        
        domain  = [ (-l,l) , (-l,l) , (-l,l) ]#  
                
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
        
        offset = 2
        
        # ground line
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-10,0,0])
        pts[1,:] = np.array([+10,0,0])
        
        lines_pts.append( pts )
        
        # pendulum no 1
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([- offset,0,0])
        
        [c1,s1,c2,s2] = self.trig( q )
        
        pts[1,:] = np.array([ s1 * self.l1 - offset , - c1 * self.l1 ,0])
        
        lines_pts.append( pts )
        
        # pendulum no 1
        
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([offset,0,0])
        
        [c1,s1,c2,s2] = self.trig( q )
        
        pts[1,:] = np.array([ s2 * self.l1 + offset , - c2 * self.l1 ,0])
        
        lines_pts.append( pts )
                
        return lines_pts
        
        
        
        
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    sys = DoublePendulum()
    
    #sys = SinglePendulum()
    
    sys = TwoIndependentSinglePendulum()
    
    def t2u(t):
        return np.array([ t + sys.u_lb[0] , t + sys.u_lb[1]])
    
    sys.t2u   = t2u
    sys.ubar[0] = 2
    sys.ubar[1] = 1
    sys.x0[0] = 3.14
    sys.compute_trajectory( 10 )
    sys.plot_trajectory('xu')
    sys.animate_simulation()