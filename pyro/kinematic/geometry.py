#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:12:12 2023

@author: alex
"""

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
###############################################################################


###############################################################################

# Notation

# aRb : rotation matrix of basis b in basis a
# ATB : transformation matrix of frame b in frame a
# v_a : vector components in basis a


###############################################################################
def transformation_matrix_2D( theta , x , y ):
    """
    
    Transformation Matrix between 2 frame in 2D

    Parameters
    ----------
    theta : float
        roation angle (arround z)
    x : float
        translation along x-axis
    y : float
        translation along y-axis

    Returns
    -------
    T : 3 x 3 np.array
        Transformation matrix

    """
    
    s = np.sin( theta )
    c = np.cos( theta )
    
    T = np.array([ [ c   , -s ,  x ] , 
                   [ s   ,  c ,  y ] ,
                   [ 0   ,  0 ,  1 ] ])
    
    return T