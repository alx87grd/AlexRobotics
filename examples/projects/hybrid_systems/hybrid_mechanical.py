#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:46:06 2022
@author: alex
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import system
from pyro.dynamic import mechanical
from hybrid import SwitchedSystem
###############################################################################

###############################################################################
        
class HybridMechanicalSystem( mechanical.MechanicalSystem , SwitchedSystem ):
    """ 
    Mechanical system with Equation of Motion in the form of
    -------------------------------------------------------
    H(q,k) ddq + C(q,dq) dq + d(q,dq,k) + g(q) = B(q,k) e
    -------------------------------------------------------
    u      :  dim = (m+1, 1)   : mode + force input variables
    k=u[0] :  integer          : operating mode
    e = u[1,:] :  dim = (m, 1) : force/torque input variables  
    q      :  dim = (dof, 1)   : position variables 
    dq     :  dim = (dof, 1)   : velocity variables     
    ddq    :  dim = (dof, 1)   : acceleration variables
    H(q,k)   :  dim = (dof, dof) : inertia matrix
    C(q)   :  dim = (dof, dof) : corriolis matrix
    B(q,k)   :  dim = (dof, m)   : actuator matrix
    ddq    :  dim = (dof, 1)   : acceleration variables
    d(q,dq,k):  dim = (dof, 1)   : state-dependent dissipative forces
    g(q)   :  dim = (dof, 1)   : state-dependent conservatives forces
    
    """
    
    ############################
    def __init__(self, dof = 1 , actuators = None, k = 2):
        """ """
        
        # Degree of Freedom
        self.dof = dof
        self.k   = k
        
        # Nb of actuators
        if actuators == None:   # If not specifyied the sys is fully actuated
            actuators = dof
        
        # Dimensions
        n = dof * 2 
        m = actuators + 1
        p = dof * 2
        
        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, n, m, p)
        
        # Name
        self.name = str(dof) + 'DoF Hybrid Mechanical System'
        
        # Labels, bounds and units
        for i in range(dof):
            # joint angle states
            self.x_ub[i] = + np.pi * 2
            self.x_lb[i] = - np.pi * 2
            self.state_label[i] = 'Angle '+ str(i)
            self.state_units[i] = '[rad]'
            # joint velocity states
            self.x_ub[i+dof] = + np.pi * 2
            self.x_lb[i+dof] = - np.pi * 2
            self.state_label[i+dof] = 'Velocity ' + str(i)
            self.state_units[i+dof] = '[rad/sec]'
        for i in range(actuators):
            self.u_ub[i+1] = + 5
            self.u_lb[i+1] = - 5
            self.input_label[i+1] = 'Torque ' + str(i)
            self.input_units[i+1] ='[Nm]'
        self.output_label = self.state_label
        self.output_units = self.state_units
        
        self.u_ub[0] = k - 1
        self.u_lb[0] = 0
        self.input_label[0] = 'Mode'
        self.input_units[0] =''
            
    ###########################################################################
    # The following functions needs to be overloaded by child classes
    # to represent the dynamic of the system
    ###########################################################################
    
    ###########################################################################
    def H(self, q , k ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        if k == 0:
            H = np.diag( np.ones( self.dof ) ) * 100 # Default is identity matrix
        else:
            H = np.diag( np.ones( self.dof ) )      # Default is identity matrix

        return H
    
    
    ###########################################################################
    def B(self, q , k ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.zeros( ( self.dof , self.m ) )
        
        if k == 0:
            for i in range(min(self.m,self.dof)):
                B[i,i] = 10                # Diag matrix for the first m rows
        else:
            for i in range(min(self.m,self.dof)):
                B[i,i] = 1                # Diag matrix for the first m rows
        
        return B
    
        
    ###########################################################################
    def d(self, q , dq , k ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        if k == 0:
            D = np.ones( self.dof ) * 100 # Default is zero vector
        else:
            D = np.zeros(self.dof ) # Default is zero vector
            
        d = np.dot( D , dq )
        
        return d
    
    
    ###########################################################################
    # No need to overwrite the following functions for custom system
    ###########################################################################
    
    ##############################
    def u2k(self, u ):
        """ Compute mode k based on u vector """  

        return int(u[0])
    
    
    ##############################
    def u2e(self, u ):
        """ Compute actuator effort e based on u vector """  
        
        return u[1:]
    
    
    ##############################
    def generalized_forces(self, q  , dq  , ddq , k , t = 0 ):
        """ Computed generalized forces given a trajectory """  
        
        H = self.H( q , k )
        C = self.C( q , dq )
        g = self.g( q )
        d = self.d( q , dq , k )
                
        # Generalized forces
        forces = np.dot( H , ddq ) + np.dot( C , dq ) + g + d
        
        return forces
    
    
    ##############################
    def actuator_forces(self, q  , dq  , ddq , k , t = 0 ):
        """ Computed actuator forces given a trajectory (inverse dynamic) """  
        
        if self.dof == (self.m - 1):
        
            B = self.B( q , k )
                    
            # Generalized forces
            forces = self.generalized_forces( q , dq , ddq , k , t )
            
            # Actuator forces
            u = np.dot( np.linalg.inv( B ) , forces )
            
            return u
        
        else:
            
            raise NotImplementedError
            
    
    ##############################
    def ddq(self, q , dq , u , t = 0 ):
        """ Computed accelerations given actuator forces (foward dynamic) """  
        
        k = self.u2k(u)
        e = self.u2e(u)
        
        H = self.H( q , k )
        C = self.C( q , dq )
        g = self.g( q  )
        d = self.d( q , dq , k )
        B = self.B( q , k )
        
        ddq = np.dot( np.linalg.inv( H ) ,  ( np.dot( B , e )  
                                            - np.dot( C , dq ) - g - d ) )
        
        return ddq
    
    
    
    ###########################################################################
    def kinetic_energy(self, q  , dq , k ):
        """ Compute kinetic energy of manipulator """  
        
        e_k = 0.5 * np.dot( dq , np.dot( self.H( q , k ) , dq ) )
        
        return e_k
    

##############################################################################
# Hybrid Mechanica lSystem
##############################################################################

class MultipleSpeedMechanicalSystem( HybridMechanicalSystem ):
    """ 
    Mechanical system with Equation of Motion in the form of
    -------------------------------------------------------
    [ H(q) + R(k)^T I R(k) ] ddq + C(q,dq) dq + [d_mech(q,dq) + R(k)^T B R(k) dq]  + g(q) = R(k)^T e
    -------------------------------------------------------
    u           :  dim = (m+1, 1)   : mode + force input variables
    k=u[0]      :  integer          : operating mode
    e = u[1,:]  :  dim = (m, 1)     : force/torque input variables  
    q           :  dim = (dof, 1)   : position variables 
    dq          :  dim = (dof, 1)   : velocity variables     
    ddq         :  dim = (dof, 1)   : acceleration variables
    H_mech(q)   :  dim = (dof, dof) : mechanism inertia matrix
    I_actuator  :  dim = (dof, dof) : actuator inertia matrix
    B_actuator  :  dim = (dof, dof) : actuator damping matrix
    C(q)        :  dim = (dof, dof) : corriolis matrix
    R           :  dim = (dof, m)   : actuator matrix
    ddq         :  dim = (dof, 1)   : acceleration variables
    d_mech(q,dq):  dim = (dof, 1)   : state-dependent dissipative forces
    g(q)        :  dim = (dof, 1)   : state-dependent conservatives forces
    
    """
    
    ############################
    def __init__(self, dof = 1 , k = 2):
        """ """
        
        super().__init__(dof, dof, k)
        
        # Name
        self.name = str(dof) + ' DoF Multiple Speed Mechanical System'
        
        # Number of discrete modes
        self.k = k  
        
        # Actuator
        self.I_actuators = np.diag( np.ones( self.dof ) )
        self.B_actuators = np.diag( np.ones( self.dof ) )
        
        # Transmissions
        self.R_options = [ np.diag( np.ones( self.dof ) ) , 
                           np.diag( np.ones( self.dof ) ) * 10 ]
        
        
            
    ###########################################################################
    # The following functions needs to be overloaded by child classes
    # to represent the dynamic of the system
    ###########################################################################
    
    ###########################################################################
    def H_mech(self, q ):
        """ 
        Inertia matrix of arm mechanism only
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        """  
        
        H = np.diag( np.ones( self.dof ) ) # Default is identity matrix
        
        return H
    
        
    ###########################################################################
    def H(self, q , k ):
        """   """  
        
        R = self.R_options[k]
        
        H = self.H_mech(q) + R.T @ self.I_actuators @ R
        
        return H
    
    
    ###########################################################################
    def B(self, q , k ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        R = self.R_options[k]
        
        return R.T
    
    ###########################################################################
    def d_mech(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d = np.zeros( self.dof ) 
        
        return d
    
        
    ###########################################################################
    def d(self, q , dq , k ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        R = self.R_options[k]
        
        d = self.d_mech(q, dq) + R.T @ self.B_actuators @ R @ dq
        
        return d




###########################################################################
class TwoSpeedLinearActuator( HybridMechanicalSystem ):
    """ 
    
    
    """
    
    ############################
    def __init__(self):
        """ """
        
        super().__init__( 1 , 1 )
        
        self.mass_output   = 10
        
        self.motor_inertia = 1
        
        self.ratios = np.array([1,10])
        
        self.motor_damping = 1
        
        self.l2 = 1
            
    ###########################################################################
    # The following functions needs to be overloaded by child classes
    # to represent the dynamic of the system
    ###########################################################################
    
    ###########################################################################
    def H(self, q , k ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        H = np.array([[ self.mass_output + self.motor_inertia * self.ratios[k]**2 ]])

        return H
    
    
    ###########################################################################
    def B(self, q , k ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.array([[ self.ratios[k] ]])
        
        return B
    
        
    ###########################################################################
    def d(self, q , dq , k ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        D = np.array([[ self.motor_damping * self.ratios[k]**2 ]])
        
        d = np.dot( D , dq )
        
        return d
    
    
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

        # mass
        pts      = np.zeros(( 5 , 3 ))
        
        
        pts[0,:] =  np.array([q[0] - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([q[0] + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([q[0] + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([q[0] - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
                
        return lines_pts , lines_style , lines_color
    
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    
    sys = MultipleSpeedMechanicalSystem(2,2)
    
    sys.x0[1] = 0
    
    sys.ubar[0] = 1
    sys.ubar[1] = 5
    
    sys.compute_trajectory()
    
    sys.plot_trajectory('xu')
    
    