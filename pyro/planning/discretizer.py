# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:02:12 2017

@author: alxgr
"""

import numpy as np

'''
################################################################################
'''


class GridDynamicSystem:
    """ Create a discrete gird state-action space for a continous dynamic system """
    
    ############################
    def __init__(self, sys , xgriddim = ( 101 , 101 ), ugriddim = ( 11 , 1 ) , dt = 0.05 ):
        
        self.sys = sys # Dynamic system class
        
        # Discretization Parameters
        
        self.dt    = dt        # time discretization
        
        # Grid size
        self.xgriddim = xgriddim
        self.ugriddim = ugriddim
        
        # Options
        self.uselookuptable = True
        
        self.compute()  
        
    ##############################
    def compute(self):
        """  """

        self.discretizespace()
        self.discretizeactions() 
        
        print('\nDiscretization:\n---------------------------------')
        print('State space dimensions:', self.sys.n , ' Input space dimension:', self.sys.m )
        print('Number of nodes:', self.nodes_n , ' Number of actions:', self.actions_n )
        print('Number of node-action pairs:', self.nodes_n * self.actions_n )
        
        self.generate_nodes()
        self.generate_actions()
        
        if self.uselookuptable:
            self.compute_lookuptable()
            
        
    #############################
    def discretizespace(self):
        """ Grid the state space """
                        
        self.xd       = []
        self.nodes_n  = 1
        
        # n-D grid
        self.x_grid2node    = np.zeros( self.xgriddim , dtype = int )     # grid of corresponding index
        
        # linespace for each x-axis and total number of nodes
        for i in range(self.sys.n):
            self.xd.append(  np.linspace( self.sys.x_lb[i]  , self.sys.x_ub[i]  , self.xgriddim[i]  ) )
            self.nodes_n        = self.nodes_n * self.xgriddim[i]
        
        # 1-D List of nodes
        self.nodes_state    = np.zeros(( self.nodes_n , self.sys.n ), dtype = float )  # Number of nodes x state dimensions
        self.nodes_index    = np.zeros(( self.nodes_n , self.sys.n ), dtype = int   )  # Number of nodes x state dimensions
        
        
    #############################
    def discretizeactions(self):
        """ Grid the action space """
        
        self.ud         = []
        self.actions_n  = 1
        
        # linespace for each u-axis and total number of actions
        for i in range(self.sys.m):
            self.ud.append(  np.linspace( self.sys.u_lb[i]  , self.sys.u_ub[i]  , self.ugriddim[i]  ) )
            self.actions_n       = self.actions_n * self.ugriddim[i]
        
        # 1-D List of actions
        self.actions_input     =   np.zeros(( self.actions_n , self.sys.m ), dtype = float )  # Number of actions x inputs dimensions
        self.actions_index     =   np.zeros(( self.actions_n , self.sys.m ), dtype = int   )  # Number of actions x inputs dimensions
        
        
    ##############################
    def generate_nodes(self):
        """ Compute 1-D list of nodes """
        
        # For all state nodes
        node = 0
        
        if self.sys.n == 2 :
            
            for i in range(self.xgriddim[0]):
                for j in range(self.xgriddim[1]):
                    
                    # State
                    x = np.array([ self.xd[0][i]  ,  self.xd[1][j] ])
                    
                    # State and grid index based on node #
                    self.nodes_state[node,:] = x
                    self.nodes_index[node,:] = np.array([i,j])
                    
                    # Node # based on index ij
                    self.x_grid2node[i,j]    = node
    
                    # Increment node number
                    node = node + 1
                    
                    
        elif self.sys.n == 3:
            
            for i in range(self.xgriddim[0]):
                for j in range(self.xgriddim[1]):
                    for k in range(self.xgriddim[2]):
                    
                        # State
                        x = np.array([ self.xd[0][i]  ,  self.xd[1][j]  , self.xd[2][k] ])
                        
                        # State and grid index based on node #
                        self.nodes_state[node,:] = x
                        self.nodes_index[node,:] = np.array([i,j,k])
                        
                        # Node # based on index ijk
                        self.x_grid2node[i,j,k]    = node
        
                        # Increment node number
                        node = node + 1
                        
                        
                        
        elif self.sys.n == 4:
            
            # NOT yet validated!!!
            
            for i in range(self.xgriddim[0]):
                for j in range(self.xgriddim[1]):
                    for k in range(self.xgriddim[2]):
                        for l in range(self.xgriddim[3]):
                    
                            # State
                            x = np.array([ self.xd[0][i]  ,  self.xd[1][j]  , self.xd[2][k] , self.xd[3][l]])
                            
                            # State and grid index based on node #
                            self.nodes_state[node,:] = x
                            self.nodes_index[node,:] = np.array([i,j,k,l])
                            
                            # Node # based on index ijkl
                            self.x_grid2node[i,j,k,l]    = node
            
                            # Increment node number
                            node = node + 1
                    
        else:
            
            raise NotImplementedError
            
                
    ##############################
    def generate_actions(self):
        """ Compute 1-D list of actions """
        
        # For all state nodes
        action = 0
        
        # Single input
        
        if self.sys.m == 1 :
        
            for k in range(self.ugriddim[0]):
                    
                u = np.array([ self.ud[0][k] ])
                
                # State and grid index based on node #
                self.actions_input[action,:] = u
                self.actions_index[action,:] = k
    
                # Increment node number
                action = action + 1
                
        elif self.sys.m == 2 :
            
            for k in range(self.ugriddim[0]):
                for l in range(self.ugriddim[1]):
                    
                    u = np.array([ self.ud[0][k] , self.ud[1][l] ])
                    
                    # State and grid index based on node #
                    self.actions_input[action,:] = u
                    self.actions_index[action,:] = np.array([k,l])
        
                    # Increment node number
                    action = action + 1
        
        else:
            
            raise NotImplementedError
            
            
    ##############################
    def compute_lookuptable(self):
        """ Compute lookup table for faster evaluation """

        if self.uselookuptable:
            # Evaluation lookup tables      
            self.action_isok   = np.zeros( ( self.nodes_n , self.actions_n ) , dtype = bool )
            self.x_next        = np.zeros( ( self.nodes_n , self.actions_n , self.sys.n ) , dtype = float ) # lookup table for dynamic
            
            # For all state nodes        
            for node in range( self.nodes_n ):  
                
                    x = self.nodes_state[ node , : ]
                
                    # For all control actions
                    for action in range( self.actions_n ):
                        
                        u = self.actions_input[ action , : ]
                        
                        # Compute next state for all inputs
                        x_next = self.sys.f( x , u ) * self.dt + x
                        
                        # validity of the options
                        x_ok = self.sys.isavalidstate(x_next)
                        u_ok = self.sys.isavalidinput(x,u)
                        
                        self.x_next[ node,  action, : ] = x_next
                        self.action_isok[ node, action] = ( u_ok & x_ok )
                
                



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    pass