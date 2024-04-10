# -*- coding: utf-8 -*-

###############################################################################
import numpy as np
###############################################################################
from pyro.control.controller import StaticController
###############################################################################

class stable_baseline3_controller( StaticController ) :
    """ 
    Wrap a stable baseline 3 model to use it as a pyro controller
    """

    def __init__(self, model):

        self.model = model

        # Dimensions
        self.k = model.observation_space.shape[0]
        self.m = model.action_space.shape[0]
        self.p = model.observation_space.shape[0]
        
        StaticController.__init__( self, self.k, self.m, self.p)

        self.name = "Stable Baseline 3 Controller"


    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c( y, r, t)
        
        INPUTS
        y  : sensor signal vector          p x 1
        r  : reference signal vector       k x 1
        t  : time                          1 x 1
        
        OUTPUTS
        u  : control inputs vector         m x 1
        
        """
        
        u, _x = self.model.predict( y , deterministic = True )
        
        return u

        