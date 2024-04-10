# -*- coding: utf-8 -*-
"""
Created on March 20 2020

@author: Pierre
"""
###############################################################################
import numpy as np
import math
###############################################################################
from wcrt import WCRT 

from pyro.control import controller

##############################################################################

class AdaptativeController_WCRT( controller.DynamicController ):
    """ 
    Adaptative Controller for fully actuated mechanical systems (single pendulum)
    """
    
    
    ############################
    def __init__( self , model , traj = None ):
        """ """
        
        self.name = 'Adaptive controller'
        
        # Params
        self.A = np.zeros(8)
        self.T=np.eye(8)
        self.Kd = np.eye(3)
        self.lam  = 1   # Sliding surface slope
        self.nab  = 0.1 # Min convergence rate
        
        self.model=model
        
        k = model.dof   
        m = model.m
        p = model.p
        l = self.A.shape[0]
        
        super().__init__(k, l, m, p) 
        
        
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
        c3  = np.cos( q[2] )
        s3  = np.sin( q[2] )
        c23 = np.cos( q[2] + q[1] )
        s23 = np.sin( q[2] + q[1] )
        
        return [c1,s1,c2,s2,c3,s3,c23,s23]

        
    ############################
    def adaptative_variables( self , ddq_d , dq_d , q_d , dq , q ):
        """ 
        
        Given desired trajectory and actual state
        
        """        
        q_e   = q  -  q_d
        dq_e  = dq - dq_d
        
        s      = dq_e  + self.lam * q_e
        dq_r   = dq_d  - self.lam * q_e
        ddq_r  = ddq_d - self.lam * dq_e
        
        return [ s , dq_r , ddq_r ]
        
        
    ############################
    def adaptative_torque( self , Y , s , q , t ):
        """ 
        
        Given actual state, compute torque necessarly to guarantee convergence
        
        """
                
        u_computed      = np.dot( Y , self.A  )
        
        u_discontinuous = np.dot(self.Kd,s)
        
        u_tot = u_computed - u_discontinuous
        
        return u_tot
    
                        
    ############################
    def b(self, z, x, q_d, t):
        
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        
        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )
        
        [c1,s1,c2,s2,c3,s3,c23,s23] = self.trig( q )
        
        Y = np.zeros((3,8))
        dz = np.zeros(8)
        
        Y[0,0]=ddq_r[0]
        Y[0,1]=ddq_r[1]*(s2+s2*c3)
        Y[0,2]=ddq_r[2]*s23
        Y[0,3]=(dq[2]*s2*s3+dq[1]*c2+dq[1]*c2*c3)*dq_r[1]
        Y[0,4]=(dq[2]+dq[1])*c23*dq_r[2]
        Y[0,5]=0
        Y[0,6]=0
        Y[0,7]=q[0]
        Y[1,0]=ddq_r[0]*(s2+s2*c3)
        Y[1,1]=ddq_r[1]*(1+c3+c3**2)
        Y[1,2]=ddq_r[2]*(c3+c3**2)
        Y[1,3]=(dq[2]*(s3+s3*c3))*dq_r[1]
        Y[1,4]=(dq[2]*(s3+s3*c3)+dq[1]*(s3+s3*c3)+dq[0]*(c2*c3))*dq_r[2]
        Y[1,5]=c2
        Y[1,6]=c23
        Y[1,7]=q[1]
        Y[2,0]=ddq_r[0]*s23
        Y[2,1]=ddq_r[1]*(c3+c3**2)
        Y[2,2]=ddq_r[2]
        Y[2,3]=(dq[1]*(s3+s3*c3)+dq[1]*c2*c3)*dq_r[1]
        Y[2,4]=0
        Y[2,5]=0
        Y[2,6]=c23
        Y[2,7]=q[2]
        
        b = np.dot(Y.T,s)
        dz=-1*np.dot( self.T , b )
            
        return dz
    
    
    ############################
    def c( self , z , x , q_d , t = 0 ):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        
        [ q , dq ]     = self.model.x2q( x )
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        
        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )
        [c1,s1,c2,s2,c3,s3,c23,s23] = self.trig( q )
        
        Y = np.zeros((3,8))
                
        Y[0,0]=ddq_r[0]
        Y[0,1]=ddq_r[1]*(s2+s2*c3)
        Y[0,2]=ddq_r[2]*s23
        Y[0,3]=(dq[2]*s2*s3+dq[1]*c2+dq[1]*c2*c3)*dq_r[1]
        Y[0,4]=(dq[2]+dq[1])*c23*dq_r[2]
        Y[0,5]=0
        Y[0,6]=0
        Y[0,7]=q[0]
        Y[1,0]=ddq_r[0]*(s2+s2*c3)
        Y[1,1]=ddq_r[1]*(1+c3+c3**2)
        Y[1,2]=ddq_r[2]*(c3+c3**2)
        Y[1,3]=(dq[2]*(s3+s3*c3))*dq_r[1]
        Y[1,4]=(dq[2]*(s3+s3*c3)+dq[1]*(s3+s3*c3)+dq[0]*(c2*c3))*dq_r[2]
        Y[1,5]=c2
        Y[1,6]=c23
        Y[1,7]=q[1]
        Y[2,0]=ddq_r[0]*s23
        Y[2,1]=ddq_r[1]*(c3+c3**2)
        Y[2,2]=ddq_r[2]
        Y[2,3]=(dq[1]*(s3+s3*c3)+dq[1]*c2*c3)*dq_r[1]
        Y[2,4]=0
        Y[2,5]=0
        Y[2,6]=c23
        Y[2,7]=q[2]
        
        self.A =self.get_z_integral( z )
                         
        u                     = self.adaptative_torque( Y , s  , q , t )
        
        return u
    

    ############################
    def get_z_integral(self, z):
        """ get intergral error internal states """
        
        return z[:self.l]
    


pi = math.pi

sys = WCRT()
sys.cost_function = None
ctl  = AdaptativeController_WCRT(sys)

#Param Wcrt
sys.d1 = 3
sys.d2 = 2
sys.d3 = 2

sys.k1 = 10
sys.k2 = 10
sys.k3 = 10

sys.m1 = 2
sys.m2 = 1
sys.m3 = 1

sys.l1  = 0.5 
sys.l2  = 0.8
sys.l3  = 0.8
sys.lc1 = 0.4
sys.lc2 = 0.5
sys.lc3 = 0.7

#Param adapt-control
ctl.z0[0] = 5
ctl.z0[1] = 5
ctl.z0[2] = 5
ctl.z0[3] = 0
ctl.z0[4] = 5
ctl.z0[5] = 20
ctl.z0[6] = 10
ctl.z0[7] = 0

ctl.Kd[0,0] = 7
ctl.Kd[1,1] = 7
ctl.Kd[2,2] = 7

ctl.lam = 1.5

ctl.T[0,0] = 8
ctl.T[1,1] = 8
ctl.T[2,2] = 8
ctl.T[3,3] = 8
ctl.T[4,4] = 8
ctl.T[5,5] = 8
ctl.T[6,6] = 8
ctl.T[7,7] = 8

#Set Point
ctl.rbar = np.array([0,-pi/4,pi/2])

#New cl-dynamic
cl_sys = ctl + sys
#cl_sys = sys

#Simultation
cl_sys.x0[0]  = pi/3
cl_sys.x0[1]  = 1
cl_sys.x0[2]  = 0

cl_sys.state_label[6] = 'H1'
cl_sys.state_label[7] = 'H2'
cl_sys.state_label[8] = 'H3'
cl_sys.state_label[9] = 'C2'
cl_sys.state_label[10] = 'C3'
cl_sys.state_label[11] = 'g1'
cl_sys.state_label[12] = 'g2'
cl_sys.state_label[13] = 'k'

cl_sys.compute_trajectory(tf=10, n=10001, solver='euler')
cl_sys.plot_trajectory()
cl_sys.plot_internal_controller_states()
cl_sys.animate_simulation(is_3d = True)

