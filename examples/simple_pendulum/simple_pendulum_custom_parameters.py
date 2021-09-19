# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:19:01 2018

@author: nvidia
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import pendulum
###############################################################################

###############################################################################
        
class MyCustomPendulum( pendulum.SinglePendulum ):
    """ 

    """
    ###########################################################################
    # Only overload functions that are different from base version
    ###########################################################################
    def setparams(self):
        """ Set model parameters here """
        
        # kinematic
        self.l1  = 3 
        self.lc1 = 2
        
        # dynamic
        self.m1       = 10
        self.I1       = 10
        self.gravity  = 9.81
        self.d1       = 50
        

'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    sys =  MyCustomPendulum()
    
    # Can also change param here
    
    sys.gravity = 2
    
    #
    
    sys.x0 = np.array([0.8,0])
    
    sys.plot_trajectory()
    sys.animate_simulation()