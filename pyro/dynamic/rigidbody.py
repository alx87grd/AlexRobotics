# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:45:37 2018

@author: Alexandre
"""


###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import system
from pyro.kinematic import geometry
from pyro.kinematic import drawing
###############################################################################

###############################################################################
class GeneralizedMechanicalSystem( system.ContinuousDynamicSystem ):
    """ 
    Mechanical system where the generalized velocities are not the time 
    derivative of the generalized coordinates.
    -------------------------------------------------------
    M(q) dv + C(q,v) v + d(q,v) + g(q) = B(q) u
    dq = N(q) v
    -------------------------------------------------------
    v        :  dim = (dof, 1)   : velocity variables
    q        :  dim = (pos, 1)   : position variables 
    dq       :  dim = (pos, 1)   : derivatives of position variables  
    dv       :  dim = (dof, 1)   : acceleration variables
    u        :  dim = (m, 1)     : force input variables
    d(q,v)   :  dim = (dof, 1)   : state-dependent dissipative forces
    g(q)     :  dim = (dof, 1)   : state-dependent conservatives forces
    M(q)     :  dim = (dof, dof) : inertia matrix
    C(q,v)   :  dim = (dof, dof) : corriolis matrix
    B(q)     :  dim = (dof, m)   : actuator matrix
    N(q)     :  dim = (pos, dof) : transformation matrix
    
    """
    
    ############################
    def __init__(self, dof = 1 , pos = None, actuators = None):
        """ """
        
        # Degree of Freedom
        self.dof = dof

        # Nb of configurations
        if pos == None:   # If not specifyied 
            pos = dof
        self.pos = pos
        
        # Nb of actuators
        if actuators == None:   # If not specifyied the sys is fully actuated
            actuators = dof
        
        # Dimensions
        n = dof + pos
        m = actuators
        p = n
        
        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, n, m, p)
        
        # Name
        self.name = str(dof) + 'DoF Generalized Mechanical System'
        
        # Labels, bounds and units
        for i in range(pos):
            # positions states
            self.x_ub[i] = + 10
            self.x_lb[i] = - 10
            self.state_label[i] = 'Position '+ str(i)
            self.state_units[i] = '[m]'
        for i in range(dof):
            # generalized velocity states
            j = i + pos
            self.x_ub[j] = + 10
            self.x_lb[j] = - 10
            self.state_label[j] = 'Velocity ' + str(i)
            self.state_units[j] = '[m/sec]'
        for i in range(actuators):
            self.u_ub[i] = + 5
            self.u_lb[i] = - 5
            self.input_label[i] = 'Force input ' + str(i)
            self.input_units[i] ='[N]'
        self.output_label = self.state_label
        self.output_units = self.state_units
            
    ###########################################################################
    # The following functions needs to be overloaded by child classes
    # to represent the dynamic of the system
    ###########################################################################
    
    ###########################################################################
    def M(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * v^T * H(q) * v
        
        """  
        
        M = np.diag( np.ones( self.dof ) ) # Default is identity matrix
        
        return M
    
    
    ###########################################################################
    def C(self, q , v ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        C = np.zeros( ( self.dof , self.dof ) ) # Default is zeros matrix
        
        return C
    
    ###########################################################################
    def N(self, q ):
        """ 
        Transformation matrix from generalized velocities to derivatives of
        configuration variables
        ------------------------------------
        dim( N ) = ( pos , dof )

        dq = N(q) v
        ------------------------------------
        """
        
        N = np.zeros( ( self.pos , self.dof ) )
        
        for i in range(min( self.pos, self.dof) ):
            N[i,i] = 1                # Diag matrix for the first m rows
        
        return N
    
    
    ###########################################################################
    def B(self, q ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.zeros( ( self.dof , self.m ) )
        
        for i in range(min(self.m,self.dof)):
            B[i,i] = 1                # Diag matrix for the first m rows
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        g = np.zeros( self.dof ) # Default is zero vector
        
        return g
    
        
    ###########################################################################
    def d(self, q , v ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d = np.zeros(self.dof ) # Default is zero vector
        
        return d
    
    
    ###########################################################################
    # No need to overwrite the following functions for custom system
    ###########################################################################
    
    #############################
    def x2q( self, x ):
        """ from state vector (x) to angle and speeds (q,dq) """
        
        q  = x[ 0        : self.pos ]
        v  = x[ self.pos : self.n   ]
        
        return [ q , v ]
        
        
    #############################
    def q2x( self, q , v ):
        """ from angle and speeds (q,dq) to state vector (x) """
        
        x = np.zeros( self.n )
        
        x[ 0        : self.pos ] = q
        x[ self.pos : self.n   ] = v
        
        return x
    
    
    #############################
    def xut2q( self, x , u , t ):
        """ compute configuration variables """
        
        return self.x2q(x)[0]
    
    
    ##############################
    def generalized_forces(self, q  , v  , dv , t = 0 ):
        """ Computed generalized forces given a trajectory """  
        
        M = self.M( q )
        C = self.C( q , v )
        g = self.g( q )
        d = self.d( q , v )
                
        # Generalized forces
        forces = M @ dv + C @ v + g + d
        
        return forces
    
    
    ##############################
    def actuator_forces(self, q  , v  , dv , t = 0 ):
        """ Computed actuator forces given a trajectory (inverse dynamic) """  
        
        if self.dof == self.m:
        
            B = self.B( q )
                    
            # Generalized forces
            forces = self.generalized_forces( q , v , dv , t )
            
            # Actuator forces
            u = np.linalg.inv( B ) @ forces 
            
            return u
        
        else:
            
            raise NotImplementedError
        
    
    ##############################
    def accelerations(self, q , v , u , t = 0 ):
        """ 
        Compute accelerations vector (foward dynamic) 
        given :
        - actuator forces 
        - actual position and velocities
        """  
        
        M = self.M( q )
        C = self.C( q , v )
        g = self.g( q  )
        d = self.d( q , v)
        B = self.B( q )
        
        dv = np.linalg.inv( M ) @ ( B @ u - C @ v - g - d )
        
        return dv
    
    
    ###########################################################################
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
        
        # from state vector (x) to position and velocities 
        [ q , v ] = self.x2q( x )       
        
        # compute accelerations
        dv = self.accelerations( q , v , u , t )

        # compute derivative of position varibles
        dq = self.N( q ) @ v
        
        # convert to state vector diff (dx)
        dx = self.q2x( dq , dv )        
        
        return dx
    
    
    ###########################################################################
    def kinetic_energy(self, q  , v ):
        """ Compute kinetic energy  """  

        e = 0.5 * v.T @ self.M( q ) @ v
        
        return e





###############################################################################
        
class GeneralizedMechanicalSystemWithPositionInputs( GeneralizedMechanicalSystem ):
    """ 
    Generalized Mechanical system with position inputs

    -------------------------------------------------------
    M(q) dv + C(q,v) v + d(q,v,u) + g(q) = B(q,u) e(u)
    dq = N(q) v
    -------------------------------------------------------

    numper of inputs are:
    m = m_f + m_o
    ---------------------------------------------------
    m      :   integer         : number of inputs
    m_f    :   integer         : number of force inputs
    m_o    :   integer         : number of other inputs
    u      :   dim = ( m , 1)  : vector of all input variables

    v        :  dim = (dof, 1)   : velocity variables
    q        :  dim = (pos, 1)   : position variables 
    dq       :  dim = (pos, 1)   : derivatives of position variables  
    dv       :  dim = (dof, 1)   : acceleration variables
    e        :  dim = (m_f, 1)   : force input variables
    d(q,v,u) :  dim = (dof, 1)   : state-dependent dissipative forces
    g(q)     :  dim = (dof, 1)   : state-dependent conservatives forces
    M(q)     :  dim = (dof, dof) : inertia matrix
    C(q,v)   :  dim = (dof, dof) : corriolis matrix
    B(q,u)   :  dim = (dof, m_f) : actuator matrix
    N(q)     :  dim = (pos, dof) : transformation matrix
    
    """
    
    ############################
    def __init__(self, dof = 1 , force_inputs = 1, other_inputs = 1, pos = None):
        """ """
        
        # Degree of Freedom
        self.dof = dof

        self.m_f = force_inputs
        self.m_o = other_inputs

        # Nb of configurations
        if pos == None:   # If not specifyied 
            pos = dof
        self.pos = pos
        
        # Dimensions
        n = dof + pos
        m = self.m_f + self.m_o
        p = n
        
        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, n, m, p)
        
        # Name
        self.name = str(dof) + 'DoF Generalized Mechanical System'
        
        # Labels, bounds and units
        for i in range(pos):
            # positions states
            self.x_ub[i] = + 10
            self.x_lb[i] = - 10
            self.state_label[i] = 'Position '+ str(i)
            self.state_units[i] = '[m]'
        for i in range(dof):
            # generalized velocity states
            j = i + pos
            self.x_ub[j] = + 10
            self.x_lb[j] = - 10
            self.state_label[j] = 'Velocity ' + str(i)
            self.state_units[j] = '[m/sec]'
        for i in range(self.m_f):
            self.u_ub[i] = + 5
            self.u_lb[i] = - 5
            self.input_label[i] = 'Force input ' + str(i)
            self.input_units[i] ='[N]'
        self.output_label = self.state_label
        self.output_units = self.state_units
            
    ###########################################################################
    # The following functions needs to be overloaded by child classes
    # to represent the dynamic of the system
    ###########################################################################
        
    #############################
    def u2e( self, u ):
        """  
        extract force inputs from all inputs
        """
        
        e = u[ 0 : self.m_f ] 
        
        return e
    
    ###########################################################################
    def B(self, q , u ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.zeros( ( self.dof , self.m_f ) )
        
        for i in range(min( self.m_f , self.dof )):
            B[i,i] = 1   # Diag matrix for the first m rows
        
        return B
    
        
    ###########################################################################
    def d(self, q , v , u ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d = np.zeros(self.dof ) # Default is zero vector
        
        return d
    
    
    ###########################################################################
    # No need to overwrite the following functions for custom system
    ###########################################################################
    
    ##############################
    def generalized_forces(self, q  , v  , dv , t = 0 ):
        
        raise NotImplementedError
    
    
    ##############################
    def actuator_forces(self, q  , v  , dv , t = 0 ):
        """ Computed actuator forces given a trajectory (inverse dynamic) """  
        
        raise NotImplementedError
        
    
    ##############################
    def accelerations(self, q , v , u , t = 0 ):
        """ 
        Compute accelerations vector (foward dynamic) 
        given :
        - actuator forces 
        - actual position and velocities
        """  
        
        M = self.M( q )
        C = self.C( q , v )
        g = self.g( q  )
        d = self.d( q , v,  u )
        B = self.B( q , u )

        e = self.u2e( u )
        
        dv = np.linalg.inv( M ) @ ( B @ e - C @ v - g - d )
        
        return dv
    
    

###############################################################################
        
class RigidBody2D( GeneralizedMechanicalSystemWithPositionInputs ):    
    """

    Mechanical system with Equations of Motion written in 
    a body-fixed frame of reference

    """

    ############################
    def __init__(self, force_inputs = 2, other_inputs = 0):
        """ """
        
        # Degree of Freedom
        self.dof = 3
        self.pos = 3

        self.m_f = force_inputs
        self.m_o = other_inputs

        # Dimensions
        n = 6
        m = self.m_f + self.m_o
        p = n
        
        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, n, m, p)
        
        # Name
        self.name = 'Planar Rigid Body'
        self.state_label = ['x','y','theta','v1','v2','w']
        self.state_units = ['[m]','[m]','[rad]','[m/sec]','[m/sec]','[rad/sec]']
        self.input_label = ['u1','u2']
        self.input_units = ['[]','[]']
        self.output_label = self.state_label
        self.output_units = self.state_units

        # Dynamic properties
        self.mass     = 1.0
        self.inertia  = 1.0
        self.l_t      = 1.0 # Distance between CG and Thrust

        # Default graphic output parameters
        self.dynamic_domain  = True


    ###########################################################################
    def M(self, q ):
        
        M = np.diag( np.array([ self.mass , self.mass, self.inertia ]) )
        
        return M
    
    
    ###########################################################################
    def C(self, q , v ):

        C = np.zeros( ( self.dof , self.dof ) ) 
        
        w = v[2]
        
        C[1,0] = + self.mass * w
        C[0,1] = - self.mass * w
        
        return C
    
    ###########################################################################
    def N(self, q ):
        """ 
        Transformation matrix from generalized velocities to derivatives of
        configuration variables
        ------------------------------------
        dim( N ) = ( pos , dof )

        dq = N(q) v
        ------------------------------------
        """

        theta = q[2]

        N = np.array( [ [ np.cos(theta) , -np.sin(theta)  , 0 ] ,
                        [ np.sin(theta) , +np.cos(theta)  , 0 ] ,
                        [ 0             , 0               , 1 ] ] )
        
        return N
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        g = np.zeros( self.dof ) # Default is zero vector
        
        return g
    
        
    ###########################################################################
    def d(self, q , v , u):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d = np.zeros(self.dof ) # Default is zero vector
        
        return d
    
    ###########################################################################
    def B(self, q , u ):
        """ 
        Actuator Matrix 
        ------------------
        This placeholder is for a 2D point force [ F_x , F_y ] 
        applied at a point located at a distance l_t behind the CG
        """
        
        B = np.zeros(( self.dof, self.m_f ))
        
        B[0,0] = 1
        B[1,1] = 1
        B[2,1] = - self.l_t 
        
        return B
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """
        Place holder graphical output ( box with a force )
        """

        l = self.l_t * 10
        
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
        Place holder graphical output ( box with a force )
        """
        
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

        w = self.l_t

        # Points in body frame
        pts      = np.zeros(( 5 , 3 ))
        pts[0,:] = np.array([-w,+w,0])
        pts[1,:] = np.array([-w,-w,0])
        pts[2,:] = np.array([+w,-w,0])
        pts[3,:] = np.array([+w,+w,0])
        pts[4,:] = np.array([-w,+w,0])
        
        pts_W    = drawing.transform_points_2D( W_T_B , pts )

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

        ###########################
        #  Diff flat output
        ###########################

        J = self.inertia
        m = self.mass
        r = self.l_t

        xp = x + J / (m * r) * np.cos(theta)
        yp = y + J / (m * r) * np.sin(theta)

        pts      = np.zeros(( 1 , 3 ))
        pts[0,:] = np.array([xp,yp,0])
        
        lines_pts.append( pts )
        lines_style.append( 'o')
        lines_color.append( 'r' )
                
        return lines_pts , lines_style , lines_color
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """
        Place holder graphical output ( box with a force )
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []

        # M per Newton of force
        f2r = 1.0 / self.u_ub[0] * self.l_t
        
        ###########################
        # force vector
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

        return lines_pts , lines_style , lines_color

    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    #sys = GeneralizedMechanicalSystem( dof = 2 , pos = 2 , actuators = 2 )
    #sys = GeneralizedMechanicalSystemWithPositionInputs( dof = 3 , pos = 1 , force_inputs= 1 , other_inputs=1 )
    sys = RigidBody2D()
    
    #sys.show(  q = np.array([ 1.0, 2.0, 0.5 ]) )
    
    sys.ubar = np.array([1,0.2])
    sys.x0   = np.array([0,0,0,0,0,0])
    
    sys.compute_trajectory( tf = 20 )
    sys.plot_trajectory()
    sys.animate_simulation()
        