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
from pyro.kinematic import geometry
###############################################################################


###############################################################################
### Transformation tools
###############################################################################


###############################################################################
def transform_points_2D( A_T_B , pts_B ):
    """
    
    Take a list of pts in a given frame B and express them in frame A base on 
    transformation matrix A_T_B that describe a 2D transform in the x-y plane

    Parameters
    ----------
    pts : TYPE
        DESCRIPTION.
    T : 3 x 3 numpy array
        A 2D transformation matrix
        T = np.array([ [ c   , -s ,  x ] , 
                       [ s   ,  c ,  y ] ,
                       [ 0   ,  0 ,  1 ] ])

    Returns
    -------
    pts_transformed : TYPE
        DESCRIPTION.

    """
    
    # Init output array of same dimension
    pts_A = np.zeros( pts_B.shape )
    
    # save z values
    z = pts_B[:,2]
    
    # set last component to 1 for homegeneous transform operations
    pts_B[:,2] = 1.
    
    # Transform all (x,y) pts in the list
    for i in range(pts_B.shape[0]):
        
        pts_A[ i ] = A_T_B @ pts_B[ i ]
        
    # reset original z values
    pts_A[:,2] = z
    
    return pts_A


###############################################################################
### drawing shorcuts tools
###############################################################################


###############################################################################
def arrow_from_length_angle( l , theta , x = 0 , y = 0 , w = None , origin = 'base'):

    # width of arrow secondary lines
    if w is None:
        d = l * 0.15          
    else:
        d = w
    
    # Local points

    if origin == 'tip':
        
        pts_local = np.array([ [ -l  ,  0 ,  0. ] , 
                               [ 0   ,  0 ,  0. ] ,
                               [ -d  ,  d ,  0. ] ,
                               [ 0   ,  0 ,  0. ] ,
                               [  -d , -d ,  0. ] ])
    
    # elif origin == 'base':
    else:
        
        pts_local = np.array([ [ 0   ,  0 ,  0. ] , 
                               [ l   ,  0 ,  0. ] ,
                               [ l-d ,  d ,  0. ] ,
                               [ l   ,  0 ,  0. ] ,
                               [ l-d , -d ,  0. ] ])
    
    T = geometry.transformation_matrix_2D( theta , x , y )
    
    pts_global = transform_points_2D( T , pts_local )
    
    
    return pts_global


###############################################################################
def arrow_from_components( vx , vy , x = 0 , y = 0 , w = None , origin = 'base' ):
    
    l     = np.sqrt( vx**2 + vy**2 )
    theta = np.arctan2( vy , vx )
    
    return arrow_from_length_angle( l , theta , x , y , w , origin )


