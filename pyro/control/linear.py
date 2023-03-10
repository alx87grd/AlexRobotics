# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:37:48 2018

@author: alxgr
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.control import controller
###############################################################################

###############################################################################
# Function
###############################################################################

def to_2D_arr(arr):
    arr = np.asanyarray(arr)
    if arr.ndim == 2:
        return arr

    if arr.ndim == 1:
        return arr[np.newaxis]
    elif arr.ndim == 0:
        return arr[np.newaxis, np.newaxis]
    else:
        raise ValueError(
            "Cannot expand array with %d dimensions to 2-D" % (arr.ndim)
        )



###############################################################################
# Simple proportionnal controller
###############################################################################
        
class ProportionnalSingleVariableController( controller.StaticController ) :
    """ 
    Simple proportionnal compensator
    ---------------------------------------
    r  : reference signal vector       k x 1
    y  : sensor signal vector          k x 1
    u  : control inputs vector         k x 1
    t  : time                          1 x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - y) * gain

    """
    
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################
    
    
    ############################
    def __init__(self, k = 1):
        """ """
        
        # Dimensions
        self.k = k   
        self.m = k   
        self.p = k
        
        controller.StaticController.__init__( self, self.k, self.m, self.p)
        
        # Label
        self.name = 'Proportionnal Controller'
        
        # Gains
        self.gain = 1
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal_proc vector     p x 1
        r  : reference signal_proc vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        u = np.zeros(self.m) # State derivative vector
        
        e = r - y
        
        u = self.gain * e
        
        return u


###############################################################################
# MIMO prop controller
###############################################################################
        
class ProportionalController(controller.StaticController):
    """
    General (SISO or MIMO) proportional controller
    -------------------------------------------------
    
    u = K * ( r - y )
    
    -----------------------------------------
    r  : reference signal vector       k x 1
    y  : sensor signal vector          p x 1
    u  : control inputs vector         m x 1
    -----------------------------------------
    
    """
    
    ###############################
    def __init__(self, m = 1 , p = 1 ):
        
        self.K = np.zeros((m,p))
        
        #self.K = to_2D_arr(K)
        #m = self.K.shape[0]
        #p = self.K.shape[1]
        
        controller.StaticController.__init__( self, p, m, p)
        
        self.rbar = np.zeros((self.k,))
        self.name = "%d X %d Proportional Controller" % self.K.shape
        
        self.ybar = np.zeros((self.p,))  # feedback offset
        self.ubar = np.zeros((self.m,))  # control input offset
        
    ############################
    @classmethod
    def from_matrix(cls, K):
        """  """
        
        K = to_2D_arr(K)
        
        m = K.shape[0]
        p = K.shape[1]
        
        instance = cls( m , p )
        
        instance.K = K
        
        return instance
        
        
    ##############################
    def c(self, y, r, t=0):
        """ Feedback law """
        
        delta_y = y - self.ybar
        delta_u = self.K.dot(r - delta_y)
        
        u = delta_u + self.ubar
        
        return u



###############################################################################
# MIMO PID controller
###############################################################################
class PIDController( controller.DynamicController ):
    """General (SISO or MIMO) PID controller

    Parameters
    ----------
    KP : array_like
        *m x p* Matrix of proportional controller gain
    KI : array_like
        *m x p* Matrix of integral controller gain
    KD : array_like
        *m x p* Matrix of derivative controller gain
    tau : float, optional
        Time constant of derivative filter.
    sat : float, optional
        Saturation of u signal

    Notes
    -----
    The error derivative is filtered and computed according to governing equations
    from:
    https://www.mathworks.com/help/physmod/sps/ref/filteredderivativediscreteorcontinuous.html

    """
    
    ##########################################################
    def __init__(self, KP, KI=None, KD=None, tau=3E-3 , sat = None):
        
        self.KP = to_2D_arr(KP)

        if KI is None:
            self.KI = np.zeros(self.KP.shape)
        else:
            self.KI = to_2D_arr(KI)
            if self.KI.shape != self.KP.shape:
                raise ValueError("Shape of KI does not match KP")

        if KD is None:
            self.KD = np.zeros(self.KP.shape)
        else:
            self.KD = to_2D_arr(KD)
            if self.KD.shape != self.KP.shape:
                raise ValueError("Shape of KD does not match KP")

        self.tau = tau
        self.sat = sat # saturation
        
        
        k = self.KP.shape[1]
        l = self.KP.shape[1]*2
        m = self.KP.shape[0]
        p = self.KP.shape[1]

        controller.DynamicController.__init__( self, k, l, m, p)
        
        for i in range(p):
            self.internal_state_label[i] = 'Integral of output ' + str(i)
            self.internal_state_label[i+self.p] = ('Filter state of output ' 
                                                   + str(i) )
        self.name = "PID Controller"
        
        
    #################################
    def b(self, z, y, r, t):
        """ Evaluate derivative of controller state """

        if z.shape != (self.l,):
            raise ValueError("Expected z with shape (%d,)" % self.l)
        if y.shape != (self.p,) or r.shape != (self.p,):
            raise ValueError("Expected r and y with shape (%d,)" % self.p)

        # Error
        e = r - y

        # Integrator state derivative
        dz_integral = e

        # Filtered derivative state
        z_filter  = self.get_z_filter( z )
        dz_filter = (e - z_filter) / self.tau

        dz = np.concatenate([dz_integral, dz_filter], axis=0)
        assert dz.shape == (self.l,)
        
        return dz
    
    
    #################################
    def c(self, z, y, r, t):
        if z.shape != (self.l, ):
            raise ValueError("expected z with shape (%d,)" % self.l)

        # Instantaneous error
        e = r - y

        # Error integral value
        ei = self.get_z_integral( z )

        # Error derivative value
        de = (e - self.get_z_filter(z)) / self.tau
        
        # Control Input
        u = self.KP.dot( e ) + self.KI.dot( ei ) + self.KD.dot( de )
        
        # Saturation
        if self.sat is not None:
            u = np.clip( u , -self.sat , self.sat)

        return u
    
    
    #################################
    def get_z_integral(self, z):
        """ get intergral error internal states """
        
        return z[:self.p]
    
    
    #################################
    def get_z_filter(self, z):
        """ get filter internal states """
        
        return z[self.p:]


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":
    from pyro.dynamic import integrator
    
    plant = integrator.DoubleIntegrator()
    
    ctl      = PIDController( 10 , 0.1 , 5)
    ctl.rbar = np.array([ 2 ])
    
    sys = ctl + plant
    
    
    sys.compute_trajectory()
    sys.plot_trajectory('xu')
    sys.plot_trajectory_with_internal_states('x')
    
    
    
    
    
    
    