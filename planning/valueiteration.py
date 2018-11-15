# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:09:37 2017

@author: alxgr
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as interpol2D
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator

from AlexRobotics.control import controller


'''
################################################################################
'''
class ViController( controller.StaticController ) :
    """ 
    Simple proportionnal compensator
    ---------------------------------------
    r  : reference signal vector  k x 1
    y  : sensor signal vector     k x 1
    u  : control inputs vector    k x 1
    t  : time                     1 x 1
    ---------------------------------------
    u = c( y , r , t ) 

    """
    
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################
    
    
    ############################
    def __init__(self, k , m ,  p):
        """ """
        
        # Dimensions
        self.k = k   
        self.m = m   
        self.p = p
        
        controller.StaticController.__init__(self, self.k, self.m, self.p)
        
        # Label
        self.name = 'Value Iteration Controller'
        
    
    #############################
    def vi_law( self , x ):
        """ 
                
        """
        
        u = np.zeros(self.m) # State derivative vector
        
        return u

        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        x = y
        u = self.vi_law( x )
        
        return u
    
    

class ValueIteration_2D:
    """ Dynamic programming for 2D continous dynamic system, one continuous input u """
    
    ############################
    def __init__(self, grid_sys , cost_function ):
        
        # Dynamic system
        self.grid_sys  = grid_sys         # Discretized Dynamic system class
        self.sys  = grid_sys.sys     # Base Dynamic system class
        
        # Controller
        self.ctl = ViController( 2 , 1 , 2)
        
        # Cost function
        self.cf  = cost_function
        
        
        # Options
        self.uselookuptable = True
        
        
    ##############################
    def initialize(self):
        """ initialize cost-to-go and policy """

        self.J             = np.zeros( self.grid_sys.xgriddim , dtype = float )
        self.action_policy = np.zeros( self.grid_sys.xgriddim , dtype = int   )

        self.Jnew          = self.J.copy()
        self.Jplot         = self.J.copy()

        # Initial evaluation
        
        # For all state nodes        
        for node in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.nodes_state[ node , : ]
                
                i = self.grid_sys.nodes_index[ node , 0 ]
                j = self.grid_sys.nodes_index[ node , 1 ]
                
                # Final Cost
                self.J[i,j] = self.cf.h( x )
                        
                
    ###############################
    def compute_step(self):
        """ One step of value iteration """
        
        # Get interpolation of current cost space
        J_interpol = interpol2D( self.grid_sys.xd[0] , self.grid_sys.xd[1] , self.J , bbox=[None, None, None, None], kx=1, ky=1,)
        
        # For all state nodes        
        for node in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.nodes_state[ node , : ]
                
                i = self.grid_sys.nodes_index[ node , 0 ]
                j = self.grid_sys.nodes_index[ node , 1 ]

                # One steps costs - Q values
                Q = np.zeros( self.grid_sys.actions_n  ) 
                
                # For all control actions
                for action in range( self.grid_sys.actions_n ):
                    
                    u = self.grid_sys.actions_input[ action , : ]
                    
                    # Compute next state and validity of the action                    
                    
                    if self.uselookuptable:
                        
                        x_next        = self.grid_sys.x_next[node,action,:]
                        action_isok   = self.grid_sys.action_isok[node,action]
                        
                    else:
                        
                        x_next        = self.sys.f( x , u ) * self.dt + x
                        x_ok          = self.sys.isavalidstate(x_next)
                        u_ok          = self.sys.isavalidinput(x,u)
                        action_isok   = ( u_ok & x_ok )
                    
                    # If the current option is allowable
                    if action_isok:
                        
                        J_next = J_interpol( x_next[0] , x_next[1] )
                        
                        # Cost-to-go of a given action
                        Q[action] = self.cf.g( x , u ) + J_next[0,0]
                        
                    else:
                        # Not allowable states or inputs/states combinations
                        Q[action] = self.cf.INF
                        
                        
                self.Jnew[i,j]          = Q.min()
                self.action_policy[i,j] = Q.argmin()
                
                # Impossible situation ( unaceptable situation for any control actions )
                if self.Jnew[i,j] > (self.cf.INF-1) :
                    self.action_policy[i,j]      = -1
        
        
        # Convergence check        
        delta = self.J - self.Jnew
        j_max     = self.Jnew.max()
        delta_max = delta.max()
        delta_min = delta.min()
        print('Max:',j_max,'Delta max:',delta_max, 'Delta min:',delta_min)
        
        self.J = self.Jnew.copy()
        
        
        
    ################################
    def compute_steps(self, l = 50, plot = False):
        """ compute number of step """
               
        for i in range(l):
            print('Step:',i)
            self.compute_step()
            
            
                
    ################################
    def plot_J(self):
        """ print graphic """
        
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]
        
        self.Jplot = self.J.copy()
        
        ###################    
        
        fs = 10
        
        self.fig1 = plt.figure(figsize=(4, 4),dpi=300, frameon=True)
        self.fig1.canvas.set_window_title('Cost-to-go')
        self.ax1  = self.fig1.add_subplot(1,1,1)
        
        plt.ylabel(yname, fontsize = fs)
        plt.xlabel(xname, fontsize = fs)
        self.im1 = plt.pcolormesh( self.grid_sys.xd[0] , self.grid_sys.xd[1] , self.Jplot.T )
        plt.axis([self.sys.x_lb[0] , self.sys.x_ub[0], self.sys.x_lb[1] , self.sys.x_ub[1]])
        plt.colorbar()
        plt.grid(True)
        plt.tight_layout()
        
    
    ################################
    def plot_policy(self, i = 0 ):
        """ print graphic """
        
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]
        
        policy_plot = self.u_policy_grid[i].copy()
        
        ###################    
        
        fs = 10
        
        self.fig1 = plt.figure(figsize=(4, 4),dpi=300, frameon=True)
        self.fig1.canvas.set_window_title('Policy for u[%i]'%i)
        self.ax1  = self.fig1.add_subplot(1,1,1)
        
        plt.ylabel(yname, fontsize = fs)
        plt.xlabel(xname, fontsize = fs)
        self.im1 = plt.pcolormesh( self.grid_sys.xd[0] , self.grid_sys.xd[1] , policy_plot.T )
        plt.axis([self.sys.x_lb[0] , self.sys.x_ub[0], self.sys.x_lb[1] , self.sys.x_ub[1]])
        plt.colorbar()
        plt.grid(True)
        plt.tight_layout() 

        
    ################################
    def assign_interpol_controller(self):
        """ controller from optimal actions """
        
        # Compute grid of u
        self.u_policy_grid    = [ None ]
        self.u_policy_grid[0] = np.zeros( self.grid_sys.xgriddim , dtype = float )
        
        # For all state nodes        
        for node in range( self.grid_sys.nodes_n ):  
            
                i = self.grid_sys.nodes_index[ node , 0 ]
                j = self.grid_sys.nodes_index[ node , 1 ]
                
                if ( self.action_policy[i,j] == -1 ):
                    self.u_policy_grid[0][i,j] = 0 
                    
                else:
                    self.u_policy_grid[0][i,j] = self.grid_sys.actions_input[ self.action_policy[i,j] , 0 ]
        

        # Compute Interpol function
        self.x2u0 = interpol2D( self.grid_sys.xd[0] , self.grid_sys.xd[1] , self.u_policy_grid[0] , bbox=[None, None, None, None], kx=1, ky=1,)
        
        # Asign Controller
        self.ctl.vi_law = self.vi_law
        
        
        
    ################################
    def vi_law(self, x , t = 0 ):
        """ controller from optimal actions """
        
        u = np.zeros( self.sys.m )
        
        u[0] = self.x2u0( x[0] , x[1] )
        
        return u
        
    ################################
    def load_data(self, name = 'DP_data'):
        """ Save optimal controller policy and cost to go """
        
        try:

            self.J              = np.load( name + '_J'  + '.npy' )
            self.action_policy  = np.load( name + '_a'  + '.npy' ).astype(int)
            
        except:
            
            print('Failed to load DP data ' )
        
        
    ################################
    def save_data(self, name = 'DP_data'):
        """ Save optimal controller policy and cost to go """
        
        np.save( name + '_J'  , self.J                        )
        np.save( name + '_a'  , self.action_policy.astype(int))
        
        
        
        
        
        
        
        
        
'''
################################################################################
'''


class ValueIteration_3D:
    """ Dynamic programming for 3D continous dynamic system, 2 continuous input u """
    
    ############################
    def __init__(self, grid_sys , cost_function ):
        
        # Dynamic system
        self.grid_sys = grid_sys        # Discretized Dynamic system class
        self.sys  = grid_sys.DS     # Base Dynamic system class
        
        # Cost function
        self.cf  = cost_function
        
        # Options
        self.uselookuptable = False
        
        
    ##############################
    def initialize(self):
        """ initialize cost-to-go and policy """

        self.J             = np.zeros( self.grid_sys.xgriddim , dtype = float )
        self.J_1D          = np.zeros( self.grid_sys.nodes_n  , dtype = float )
        self.action_policy = np.zeros( self.grid_sys.xgriddim , dtype = int   )

        self.Jnew          = self.J.copy()
        self.J_1D_new      = self.J_1D.copy()
        self.Jplot         = self.J.copy()

        # Initial evaluation
        
        # For all state nodes        
        for node in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.nodes_state[ node , : ]
                
                i = self.grid_sys.nodes_index[ node , 0 ]
                j = self.grid_sys.nodes_index[ node , 1 ]
                k = self.grid_sys.nodes_index[ node , 2 ]
                
                # Final Cost
                j               = self.cf.h( x )
                self.J[i,j,k]   = j
                self.J_1D[node] = j
                        
                
    ###############################
    def compute_step(self):
        """ One step of value iteration """
        
        # Get interpolation of current cost space
        #J_interpol = interpol2D( self.grid_sys.xd[0] , self.grid_sys.xd[1] , self.J , bbox=[None, None, None, None], kx=1, ky=1,)
        
        cartcoord   = self.grid_sys.nodes_state
        values      = self.J_1D
        J_interpol  = LinearNDInterpolator(cartcoord, values, fill_value=0)
        
        
        # For all state nodes        
        for node in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.nodes_state[ node , : ]
                
                i = self.grid_sys.nodes_index[ node , 0 ]
                j = self.grid_sys.nodes_index[ node , 1 ]
                k = self.grid_sys.nodes_index[ node , 3 ]

                # One steps costs - Q values
                Q = np.zeros( self.grid_sys.actions_n  ) 
                
                # For all control actions
                for action in range( self.grid_sys.actions_n ):
                    
                    u = self.grid_sys.actions_input[ action , : ]
                    
                    # Compute next state and validity of the action                    
                    x_next        = self.sys.f( x , u ) * self.dt + x
                    x_ok          = self.sys.isavalidstate(x_next)
                    u_ok          = self.sys.isavalidinput(x,u)
                    action_isok   = ( u_ok & x_ok )
                    
                    # If the current option is allowable
                    if action_isok:
                        
                        J_next = J_interpol( x_next )
                        
                        # Cost-to-go of a given action
                        Q[action] = self.cf.g( x , u ) + J_next[0,0]
                        
                    else:
                        # Not allowable states or inputs/states combinations
                        Q[action] = self.cf.INF
                        
                        
                self.Jnew[i,j,k]          = Q.min()
                self.J_1D_new[node]       = self.Jnew[i,j,k]
                self.action_policy[i,j,k] = Q.argmin()
                
                # Impossible situation ( unaceptable situation for any control actions )
                if self.Jnew[i,j,k] > (self.cf.INF-1) :
                    self.action_policy[i,j,k]     = -1
        
        
        # Convergence check        
        delta = self.J - self.Jnew
        j_max     = self.Jnew.max()
        delta_max = delta.max()
        delta_min = delta.min()
        print('Max:',j_max,'Delta max:',delta_max, 'Delta min:',delta_min)
        
        self.J    = self.Jnew.copy()
        self.J_1D = self.J_1D_new.copy()
        
        
        
    ################################
    def compute_steps(self, l = 50, plot = False):
        """ compute number of step """
               
        for i in range(l):
            print('Step:',i)
            self.compute_step()
            
            
                
    ################################
    def plot_J_ij(self, k ):
        """ print graphic """
        
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]
        
        self.Jplot = self.J[:,:,i].copy()
        
        ###################    
        
        fs = 10
        
        self.fig1 = plt.figure(figsize=(4, 4),dpi=300, frameon=True)
        self.fig1.canvas.set_window_title('Cost-to-go')
        self.ax1  = self.fig1.add_subplot(1,1,1)
        
        plt.ylabel(yname, fontsize = fs)
        plt.xlabel(xname, fontsize = fs)
        self.im1 = plt.pcolormesh( self.grid_sys.xd[0] , self.grid_sys.xd[1] , self.Jplot.T )
        plt.axis([self.sys.x_lb[0] , self.sys.x_ub[0], self.sys.x_lb[1] , self.sys.x_ub[1]])
        plt.colorbar()
        plt.grid(True)
        plt.tight_layout()
        
    
    ################################
    def plot_policy_ij(self, k , i = 0 ):
        """ print graphic """
        
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]
        
        policy_plot = self.u_policy_grid[i][:,:,k].copy()
        
        ###################    
        
        fs = 10
        
        self.fig1 = plt.figure(figsize=(4, 4),dpi=300, frameon=True)
        self.fig1.canvas.set_window_title('Policy for u[%i]'%i)
        self.ax1  = self.fig1.add_subplot(1,1,1)
        
        plt.ylabel(yname, fontsize = fs)
        plt.xlabel(xname, fontsize = fs)
        self.im1 = plt.pcolormesh( self.grid_sys.xd[0] , self.grid_sys.xd[1] , policy_plot.T )
        plt.axis([self.sys.x_lb[0] , self.sys.x_ub[0], self.sys.x_lb[1] , self.sys.x_ub[1]])
        plt.colorbar()
        plt.grid(True)
        plt.tight_layout() 

        
    ################################
    def assign_interpol_controller(self):
        """ controller from optimal actions """
        
        # Compute grid of u
        self.u_policy_grid    = [ None ]
        self.u_policy_grid[0] = np.zeros( self.grid_sys.xgriddim , dtype = float )
        
        # For all state nodes        
        for node in range( self.grid_sys.nodes_n ):  
            
                i = self.grid_sys.nodes_index[ node , 0 ]
                j = self.grid_sys.nodes_index[ node , 1 ]
                
                if ( self.action_policy[i,j] == -1 ):
                    self.u_policy_grid[0][i,j] = 0 
                    
                else:
                    self.u_policy_grid[0][i,j] = self.grid_sys.actions_input[ self.action_policy[i,j] , 0 ]
        

        # Compute Interpol function
        self.x2u0 = interpol2D( self.grid_sys.xd[0] , self.grid_sys.xd[1] , self.u_policy_grid[0] , bbox=[None, None, None, None], kx=1, ky=1,)
        
        # Asign Controller
        self.controller.c = self.ctl_interpol
        
        
        
    ################################
    def ctl_interpol(self, x , t = 0 ):
        """ controller from optimal actions """
        
        u = np.zeros( self.sys.m )
        
        u[0] = self.x2u0( x[0] , x[1] )
        
        return u
        
    ################################
    def load_data(self, name = 'DP_data'):
        """ Save optimal controller policy and cost to go """
        
        try:

            self.J              = np.load( name + '_J'  + '.npy' )
            self.action_policy  = np.load( name + '_a'  + '.npy' ).astype(int)
            
        except:
            
            print('Failed to load DP data ' )
        
        
    ################################
    def save_data(self, name = 'DP_data'):
        """ Save optimal controller policy and cost to go """
        
        np.save( name + '_J'  , self.J                        )
        np.save( name + '_a'  , self.action_policy.astype(int))

        
        
        