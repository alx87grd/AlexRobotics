# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:09:37 2017

@author: alxgr
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as interpol2D


'''
################################################################################
'''


class ValueIteration_2D:
    """ Dynamic programming for 2D continous dynamic system, one continuous input u """
    
    ############################
    def __init__(self, dDS , cost_function ):
        
        # Dynamic system
        self.dDS = dDS        # Discretized Dynamic system class
        self.DS  = dDS.DS     # Base Dynamic system class
        
        # Cost function
        self.CF  = cost_function
        
        
        # Options
        self.uselookuptable = True
        
        
    ##############################
    def initialize(self):
        """ initialize cost-to-go and policy """

        self.J             = np.zeros( self.dDS.xgriddim , dtype = float )
        self.action_policy = np.zeros( self.dDS.xgriddim , dtype = int   )

        self.Jnew          = self.J.copy()
        self.Jplot         = self.J.copy()

        # Initial evaluation
        
        # For all state nodes        
        for node in range( self.dDS.nodes_n ):  
            
                x = self.dDS.nodes_state[ node , : ]
                
                i = self.dDS.nodes_index[ node , 0 ]
                j = self.dDS.nodes_index[ node , 1 ]
                
                # Final Cost
                self.J[i,j] = self.CF.h( x )
                        
                
    ###############################
    def compute_step(self):
        """ One step of value iteration """
        
        # Get interpolation of current cost space
        J_interpol = interpol2D( self.dDS.xd[0] , self.dDS.xd[1] , self.J , bbox=[None, None, None, None], kx=1, ky=1,)
        
        # For all state nodes        
        for node in range( self.dDS.nodes_n ):  
            
                x = self.dDS.nodes_state[ node , : ]
                
                i = self.dDS.nodes_index[ node , 0 ]
                j = self.dDS.nodes_index[ node , 1 ]

                # One steps costs - Q values
                Q = np.zeros( self.dDS.actions_n  ) 
                
                # For all control actions
                for action in range( self.dDS.actions_n ):
                    
                    u = self.dDS.actions_input[ action , : ]
                    
                    # Compute next state and validity of the action                    
                    
                    if self.uselookuptable:
                        
                        x_next        = self.dDS.x_next[node,action,:]
                        action_isok   = self.dDS.action_isok[node,action]
                        
                    else:
                        
                        x_next        = self.DS.fc( x , u ) * self.dt + x
                        x_ok          = self.DS.isavalidstate(x_next)
                        u_ok          = self.DS.isavalidinput(x,u)
                        action_isok   = ( u_ok & x_ok )
                    
                    # If the current option is allowable
                    if action_isok:
                        
                        J_next = J_interpol( x_next[0] , x_next[1] )
                        
                        # Cost-to-go of a given action
                        Q[action] = self.CF.g( x , u ) + J_next[0,0]
                        
                    else:
                        # Not allowable states or inputs/states combinations
                        Q[action] = self.CF.INF
                        
                        
                self.Jnew[i,j]          = Q.min()
                self.action_policy[i,j] = Q.argmin()
                
                # Impossible situation ( unaceptable situation for any control actions )
                if self.Jnew[i,j] > (self.CF.INF-1) :
                    self.action_policy[i,j]      = -1
        
        
        # Convergence check        
        delta = self.J - self.Jnew
        j_max     =self.Jnew.max()
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
            if plot:
                self.plot_J_update()
                
                
    ################################
    def plot_J(self):
        """ print graphic """
        
        xname = self.DS.state_label[0] + ' ' + self.DS.state_units[0]
        yname = self.DS.state_label[1] + ' ' + self.DS.state_units[1]
        
        self.Jplot = self.J
        
        ###################    
        
        fs = 10
        
        self.fig1 = plt.figure(figsize=(4, 4),dpi=300, frameon=True)
        self.fig1.canvas.set_window_title('Cost-to-go')
        self.ax1  = self.fig1.add_subplot(1,1,1)
        
        plt.ylabel(yname, fontsize = fs)
        plt.xlabel(xname, fontsize = fs)
        self.im1 = plt.pcolormesh( self.dDS.xd[0] , self.dDS.xd[1] , self.Jplot.T )
        plt.axis([self.DS.x_lb[0] , self.DS.x_ub[0], self.DS.x_lb[1] , self.DS.x_ub[1]])
        plt.colorbar()
        plt.grid(True)
        plt.tight_layout()  
                

    ################################
    def plot_J_nice(self, maxJ = 10):
        """ print graphic """
        
        xname = self.DS.state_label[0] + ' ' + self.DS.state_units[0]
        yname = self.DS.state_label[1] + ' ' + self.DS.state_units[1]
        
        ## Saturation function for cost
        for i in range(self.dDS.x0_n):
            for j in range(self.dDS.x1_n):
                if self.J[i,j] >= maxJ :
                    self.Jplot[i,j] = maxJ
                else:
                    self.Jplot[i,j] = self.J[i,j]
        
        ###################    
        
        fs = 10
        
        self.fig1 = plt.figure(figsize=(4, 4),dpi=300, frameon=True)
        self.fig1.canvas.set_window_title('Cost-to-go')
        self.ax1  = self.fig1.add_subplot(1,1,1)
        
        plt.ylabel(yname, fontsize = fs)
        plt.xlabel(xname, fontsize = fs)
        self.im1 = plt.pcolormesh( self.dDS.xd[0] , self.dDS.xd[1]  , self.Jplot.T )
        plt.axis([self.DS.x_lb[0] , self.DS.x_ub[0], self.DS.x_lb[1] , self.DS.x_ub[1]])
        plt.colorbar()
        plt.grid(True)
        plt.tight_layout()       
        
#
#        
#    ################################
#    def assign_interpol_controller(self):
#        """ controller from optimal actions """
#        
#        self.b_u0 = interpol2D( self.X[0] , self.X[1] , self.u0_policy , bbox=[None, None, None, None], kx=1, ky=1,)
#        
#        self.DS.ctl = self.feedback_law_interpol
#        
#        
#        
#    ################################
#    def feedback_law_interpol(self, x , t = 0 ):
#        """ controller from optimal actions """
#        
#        u = np.zeros( self.DS.m )
#        
#        u[0] = self.b_u0( x[0] , x[1] )
#        
#        return u
#        
#    ################################
#    def load_data(self, name = 'DP_data'):
#        """ Save optimal controller policy and cost to go """
#        
#        try:
#            
#            # Dyan prog data
#            self.X              = np.load( name + '_X'  + '.npy' )
#            self.J              = np.load( name + '_J'  + '.npy' )
#            self.action_policy  = np.load( name + '_a'  + '.npy' ).astype(int)
#            self.u0_policy      = np.load( name + '_u0' + '.npy' )
#            
#        except:
#            
#            print('Failed to load DP data ' )
#        
#        
#    ################################
#    def save_data(self, name = 'DP_data'):
#        """ Save optimal controller policy and cost to go """
#        
#        # Dyan prog data
#        np.save( name + '_X'  , self.X                        )
#        np.save( name + '_J'  , self.J                        )
#        np.save( name + '_a'  , self.action_policy.astype(int))
#        np.save( name + '_u0' , self.u0_policy                )

        
        
        