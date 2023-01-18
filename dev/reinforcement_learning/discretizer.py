# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:02:12 2017

@author: alxgr
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import time

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator

'''
################################################################################
'''


class GridDynamicSystem:
    
    ############################
    def __init__(self, sys , x_grid_dim = [ 101 , 101 ], u_grid_dim = [ 11 ] , dt = 0.05 , lookup = True ):
        """
        Class of tools for working with a discretize state space

        Parameters
        ----------
        sys : pyro ContinuousDynamicSystem class
              A dynamic system 
            
        x_grid_dim : list of integers
                     The number of discret level for each dimension of the state space 
            
        u_grid_dim : list of integers
                     The number of discret level for each dimension of the input space 
            
        dt : float
             The time step
            
        lookup : bool
                 option of computing look up table of the foward dynamics
                 
        Returns
        -------
        self.x_level : list of array
                        discrete level for each state coordinates on the grid
                        
        self.u_level : list of  array
                        discrete level for each input coordinates on the grid
                        
        self.nodes_n    : int
                          total number of discrete state on the grid
        
        self.actions_n  : int
                          total number of discrete input on the grid
                          
        self.node_id_from_index : n-D array of int
                                  The node ID based on index for each state coordinates
                                  
        self.action_id_from_index : n-D array of int
                                    The action ID based on index for each input coordinates
                                    
        self.state_from_node_id : 2-D array of float
                                  The state value based on node ID 
        
        self.index_from_node_id : 2-D array of int
                                  The indexes based on node ID 
        
        self.input_from_action_id : 2-D array of float
                                    The state value based on action ID 
        
        self.index_from_action_id : 2-D array of int
                                    The indexes based on action ID 
                                    
        more ...
        
        """
        
        # Dynamic system class
        self.sys = sys 
        
        # time discretization
        self.dt    = dt        
        
        # Grid size
        self.x_grid_dim = np.array( x_grid_dim )
        self.u_grid_dim = np.array( u_grid_dim )
        
        # Options
        self.computelookuptable = lookup
        
        # Plot params
        self.fontsize             = 5
        self.figsize              = (4, 3)
        self.dpi                  = 300
        
        # Initialize
        self.compute()  
        
        
    ##############################
    ### Initial Computations
    ##############################
        
    ##############################
    def compute(self):
        """  """

        self.discretize_state_space()
        self.discretize_input_space() 
        
        print('\nGenerating a mesh for:', self.sys.name)
        print('---------------------------------------------------')
        print('State space dimensions:', self.sys.n , ' Input space dimension:', self.sys.m )
        print('Number of nodes:', self.nodes_n , ' Number of actions:', self.actions_n )
        print('Number of node-action pairs:', self.nodes_n * self.actions_n )
        print('---------------------------------------------------')
        
        self.generate_nodes()
        self.generate_actions()
        
        if self.computelookuptable:
            self.compute_xnext_table()
            self.compute_action_set_table()
            #self.compute_nearest_snext_table() # Unused so far and long
            
        
    #############################
    def discretize_state_space(self):
        """ Grid the state space """
                        
        self.x_level  = []
        self.nodes_n  = 1
        
        # linespace for each x-axis and total number of nodes
        for i in range(self.sys.n):
            self.x_level.append(  np.linspace( self.sys.x_lb[i]  , self.sys.x_ub[i]  , self.x_grid_dim[i]  ) )
            self.nodes_n        = self.nodes_n * self.x_grid_dim[i]
        
        # range and step size for each dim
        self.x_range     = self.sys.x_ub - self.sys.x_lb
        self.x_step_size = self.x_range / ( self.x_grid_dim - 1 )
        
    #############################
    def discretize_input_space(self):
        """ Grid the input space """
        
        self.u_level    = []
        self.actions_n  = 1
        
        # linespace for each u-axis and total number of actions
        for i in range(self.sys.m):
            self.u_level.append(  np.linspace( self.sys.u_lb[i]  , self.sys.u_ub[i]  , self.u_grid_dim[i]  ) )
            self.actions_n       = self.actions_n * self.u_grid_dim[i]
        
        # range and step size for each dim
        self.u_range     = self.sys.u_ub - self.sys.u_lb
        self.u_step_size = self.u_range / ( self.u_grid_dim - 1 )
        
        
    ##############################
    def generate_nodes(self):
        """ Compute 1-D list of nodes based on a regular grid discretization """
        
        start_time = time.time()
        print('Computing nodes..  ', end = '')
        
        # n-D grid of node ID
        self.node_id_from_index = np.zeros( self.x_grid_dim , dtype = int )     # grid of node ID
        
        # 1-D List of nodes
        self.state_from_node_id = np.zeros(( self.nodes_n , self.sys.n ), dtype = float )  # Number of nodes x state dimensions
        self.index_from_node_id = np.zeros(( self.nodes_n , self.sys.n ), dtype = int   )  # Number of nodes x state dimensions
        
        # For all state nodes
        node_id = 0
        
        if self.sys.n == 2 :
            
            for i in range(self.x_grid_dim[0]):
                for j in range(self.x_grid_dim[1]):
                    
                    # State
                    x = np.array([ self.x_level[0][i]  ,  self.x_level[1][j] ])
                    
                    # State and grid index based on node id
                    self.state_from_node_id[ node_id , : ] = x
                    self.index_from_node_id[ node_id , : ] = np.array([i,j])
                    
                    # Node # based on index ij
                    self.node_id_from_index[i,j] = node_id
    
                    # Increment node number
                    node_id = node_id + 1
                    
                    
        elif self.sys.n == 3:
            
            for i in range(self.x_grid_dim[0]):
                for j in range(self.x_grid_dim[1]):
                    for k in range(self.x_grid_dim[2]):
                    
                        # State
                        x = np.array([ self.x_level[0][i]  ,  self.x_level[1][j]  , self.x_level[2][k] ])
                        
                        # State and grid index based on node #
                        self.state_from_node_id[ node_id , : ] = x
                        self.index_from_node_id[ node_id , : ] = np.array([i,j,k])
                        
                        # Node # based on index ijk
                        self.node_id_from_index[i,j,k] = node_id
        
                        # Increment node number
                        node_id = node_id + 1
                        
                        
                        
        elif self.sys.n == 4:
            
            # NOT yet validated!!!
            
            for i in range(self.x_grid_dim[0]):
                for j in range(self.x_grid_dim[1]):
                    for k in range(self.x_grid_dim[2]):
                        for l in range(self.x_grid_dim[3]):
                    
                            # State
                            x = np.array([ self.x_level[0][i]  ,  self.x_level[1][j]  , self.x_level[2][k] , self.x_level[3][l]])
                            
                            # State and grid index based on node #
                            self.state_from_node_id[ node_id , : ] = x
                            self.index_from_node_id[ node_id , : ] = np.array([i,j,k,l])
                            
                            # Node # based on index ijkl
                            self.node_id_from_index[i,j,k,l] = node_id
            
                            # Increment node number
                            node_id = node_id + 1
                    
        else:
            
            raise NotImplementedError
            
        # Print update
        computation_time = time.time() - start_time
        print('completed in %4.2f sec'%computation_time)
            
                
    ##############################
    def generate_actions(self):
        """ Compute 1-D list of actions based on a regular grid discretization"""
        
        start_time = time.time()
        print('Computing actions..  ', end = '')
        
        # m-D grid of action ID
        self.action_id_from_index = np.zeros( self.u_grid_dim , dtype = int )     # grid of node ID
        
        # 1-D List of actions
        self.input_from_action_id = np.zeros(( self.actions_n , self.sys.m ), dtype = float )  # Number of actions x inputs dimensions
        self.index_from_action_id = np.zeros(( self.actions_n , self.sys.m ), dtype = int   )  # Number of actions x inputs dimensions
        
        # For all state nodes
        action_id = 0
        
        # Single input
        
        if self.sys.m == 1 :
        
            for k in range(self.u_grid_dim[0]):
                    
                u = np.array([ self.u_level[0][k] ])
                
                # State and grid index based on node #
                self.input_from_action_id[ action_id , : ] = u
                self.index_from_action_id[ action_id , : ] = k
                
                # Action # based on index k
                self.action_id_from_index[k] = action_id
    
                # Increment node number
                action_id = action_id + 1
                
        elif self.sys.m == 2 :
            
            for k in range(self.u_grid_dim[0]):
                for l in range(self.u_grid_dim[1]):
                    
                    u = np.array([ self.u_level[0][k] , self.u_level[1][l] ])
                    
                    # State and grid index based on node #
                    self.input_from_action_id[ action_id , : ] = u
                    self.index_from_action_id[ action_id , : ] = np.array([k,l])
                    
                    # Action # based on index k
                    self.action_id_from_index[k,l] = action_id
        
                    # Increment node number
                    action_id = action_id + 1
        
        else:
            
            raise NotImplementedError
            
        # Print update
        computation_time = time.time() - start_time
        print('completed in %4.2f sec'%computation_time)
            
            
    ##############################
    def compute_action_set_table(self):
        """ Compute a boolen table describing the action set for each node """
        
        start_time = time.time()
        print('Computing action sets..  ', end = '')
            
        # Evaluation lookup tables      
        self.action_isok   = np.zeros( ( self.nodes_n , self.actions_n ) , dtype = bool )
        
        # For all state nodes        
        for node_id in range( self.nodes_n ):  
            
                x = self.state_from_node_id[ node_id , : ]
            
                # For all control actions
                for action_id in range( self.actions_n ):
                    
                    u = self.input_from_action_id[ action_id , : ]

                    u_ok = self.sys.isavalidinput(x,u)

                    self.action_isok[ node_id , action_id ] = u_ok
                    
        computation_time = time.time() - start_time
        print('completed in %4.2f sec'%computation_time)
                    
                    
    ##############################
    def compute_xnext_table(self):
        """ Compute a x_next lookup table for the forward dynamics """
        
        start_time = time.time()
        print('Computing x_next array.. ', end = '')
            
        # Evaluation lookup tables
        self.x_next_table = np.zeros( ( self.nodes_n , self.actions_n , self.sys.n ) , dtype = float ) # lookup table for dynamic
        self.x_next_isok  = np.zeros( ( self.nodes_n , self.actions_n ) , dtype = bool )
        
        # For all state nodes        
        for node_id in range( self.nodes_n ):  
            
                x = self.state_from_node_id[ node_id , : ]
            
                # For all control actions
                for action_id in range( self.actions_n ):
                    
                    u = self.input_from_action_id[ action_id , : ]
                    
                    # Compute next state for all inputs
                    x_next = self.sys.f( x , u ) * self.dt + x
                    x_ok   = self.sys.isavalidstate( x_next )
                    
                    self.x_next_table[ node_id , action_id , : ] = x_next
                    self.x_next_isok[  node_id , action_id ]     = x_ok
                
                
                if (node_id % 10000) == 9999:
                    computation_time = time.time() - start_time
                    print('\rComputing x_next array.. %d/%d nodes computed in %4.2f sec'%((node_id+1), self.nodes_n ,  computation_time ) )
                
        
        computation_time = time.time() - start_time
        print('\rComputing x_next array.. completed in %4.2f sec'%computation_time)
                    
    
    ##############################
    def compute_nearest_snext_table(self):
        """ Compute s_next lookup table for the forward dynamics """
            
        # Evaluation lookup tables
        self.s_next_table = np.zeros( ( self.nodes_n , self.actions_n ) , dtype = int ) # lookup table for dynamic
        
        # For all state nodes        
        for node_id in range( self.nodes_n ):  
            
                x = self.state_from_node_id[ node_id , : ]
            
                # For all control actions
                for action_id in range( self.actions_n ):
                    
                    # Compute the control input
                    u = self.input_from_action_id[ action_id , : ]
                    
                    # Compute next state
                    x_next = self.sys.f( x , u ) * self.dt + x
                    
                    # Compute nearest node
                    s_next = self.get_nearest_node_id_from_state( x_next )
                    
                    # Put in the lookup table
                    self.s_next_table[ node_id ,  action_id ] = s_next
                    
                    
                        
                        
    
    ##############################
    ### Quick convertion shorcut 
    ##############################
    
    ##############################
    def get_index_from_state(self, x ):
        """  
        Return state position on the grid in terms of fractionnal indexes 
        """
        
        indexes = np.zeros( self.sys.n , dtype = float )
        
        # for all state dimensions
        for i in range( self.sys.n ):
            
            indexes[i] = ( x[i] - self.sys.x_lb[i] ) / self.x_range[i] * ( self.x_grid_dim[i] - 1 )
        
        return indexes
    
    
    ##############################
    def get_nearest_index_from_state(self, x ):
        """  
        Return nearest indexes on the state-space grid from a state
        """
        
        # Round the indexes to the nearest integer
        nearest_indexes = np.rint( self.get_index_from_state( x ) ).astype(int)
        
        clipped_indexes = np.clip( nearest_indexes , 0 , self.x_grid_dim - 1 )
        
        # SHould we return -1 for out of bounds indexes??
        
        return clipped_indexes
    
    
    ##############################
    def get_nearest_node_id_from_state(self, x ):
        """  
        Return the node id that is the closest on the grid from x
        """
        
        indexes = tuple( self.get_nearest_index_from_state( x ) )
        
        node_id = self.node_id_from_index[ indexes ]
        
        return node_id
    
    
    ##############################
    def get_index_from_input(self, u ):
        """  
        Return sinput position on the grid in terms of fractionnal indexes 
        """
        
        indexes = np.zeros( self.sys.m , dtype = float )
        
        # for all state dimensions
        for i in range( self.sys.m ):
            
            indexes[i] = ( u[i] - self.sys.u_lb[i] ) / self.u_range[i] * ( self.u_grid_dim[i] - 1 )
        
        return indexes
    
    
    ##############################
    def get_nearest_index_from_input(self, u ):
        """  
        Return nearest indexes on the state-space grid from a state
        """
        
        # Round the indexes to the nearest integer
        nearest_indexes = np.rint( self.get_index_from_input( u ) ).astype(int)
        
        clipped_indexes = np.clip( nearest_indexes , 0 , self.u_grid_dim - 1 )
        
        return clipped_indexes
    
    
    ##############################
    def get_nearest_action_id_from_input(self, u ):
        """  
        Return the action id that is the closest on the grid from u
        """
        
        indexes   = tuple( self.get_nearest_index_from_input( u ) )
        
        action_id = self.action_id_from_index[ indexes ]
        
        return action_id
            
    
    ##############################
    ### Tools
    ##############################
    
    ##############################
    def get_grid_from_array(self, J ):
        """  
        convert a scalar value from node_id 1-D array to n-D array (table)
        """
        
        """
        if self.nodes_n != J.size:
            raise ValueError("Grid size does not match optimal action table size")
        
        # n-D grid of values
        J_grid = np.zeros( self.x_grid_dim , dtype = float )
        
        # For all state nodes        
        for node_id in range( self.nodes_n ): 
            
            indexes = tuple( self.index_from_node_id[ node_id , : ] )
            
            J_grid[ indexes ] = J [ node_id ]
            
        return J_grid
        """
        return J.reshape( self.x_grid_dim )
    
    ##############################
    def compute_interpolation_function(self, J , method='linear' , bounds_error = True , fill_value = None ):
        """  
        Return interpolation function for value based on x coordinates
        
        Methods: "linear”, “nearest”, “slinear”, “cubic”, and “quintic”
        """
        
        if self.nodes_n != J.size:
            raise ValueError("Grid size does not match data")
        
        # n-D grid of values
        J_grid = self.get_grid_from_array( J )
        
        levels = tuple(self.x_level[i] for i in range(self.sys.n))
        
        interpol = RegularGridInterpolator( levels , J_grid , method , bounds_error , fill_value )
        
        return interpol
    
    
    ##############################
    def compute_bivariatespline_2D_interpolation_function(self, J , kx = 1 , ky = 1 ):
        """  
        Return interpolation function for value based on x coordinates
        
        Methods: "linear”, “nearest”, “slinear”, “cubic”, and “quintic”
        """
        
        if self.sys.n == 2 : 
        
            if self.nodes_n != J.size:
                raise ValueError("Grid size does not match data")
            
            # n-D grid of values
            J_grid = self.get_grid_from_array( J )
            
            interpol = RectBivariateSpline( self.x_level[0] , self.x_level[1] , J_grid , bbox=[None, None, None, None], kx=kx, ky=ky,)
            
        else:
            
            raise NotImplementedError
        
        return interpol
    
    
    ##############################
    def get_input_from_policy(self, pi , k ):
        """  
        from pi array to k coordinate value of the control input
        """
        
        if self.nodes_n != pi.size:
            raise ValueError("Grid size does not match optimal action table size")
            
        uk_array = np.zeros( self.nodes_n , dtype = float )
        
        # For all state nodes        
        for node_id in range( self.nodes_n ): 
            
            a = pi[ node_id ]
            
            uk_array[ node_id ] = self.input_from_action_id[ a , k ]
            
        return uk_array
    
    
    ##############################
    def get_2D_slice_of_grid(self, Z, axis_1 = 0 , axis_2 = 1 ):
        
        if self.sys.n == 2:
            Z_2d = Z
        
        elif self.sys.n > 2:
            
            axis_1_dim = Z.shape[ axis_1 ]
            axis_2_dim = Z.shape[ axis_2 ]
            
            Z_2d = np.zeros( ( axis_1_dim , axis_2_dim ), dtype = float )
            
            # get defaults index for other axis
            indexes = self.get_nearest_index_from_state( self.sys.xbar )
            
            for i in range( axis_1_dim ):
                for j in range( axis_2_dim):
                    
                    indexes[ axis_1 ] = i
                    indexes[ axis_2 ] = j
                    
                    Z_2d[i,j] = Z[ tuple(indexes) ]
                    
        else:
            
            raise NotImplementedError
            
        return Z_2d
    
    
    ##############################
    def plot_grid_value(self, J , name = 'Value on the grid' , x = 0 , y = 1, jmax =  np.inf , jmin = -1 , cmap = 'YlOrRd'):
        """  
        plot a scalar value (array by node-id) on a grid
        
        Parameters
        ----------
        J : n-D numpy array
        
        name : string
               name of the figure
        
        x : int 
            index of J axis to plot as the x-axis on the graph
            
        y : int 
            index of J axis to plot as the y-axis on the graph
            
        jmax : float
               maximum value to clip the J array on the plot
            
        jmin : float
               minimum value to clip the J array on the plot
        """
        
        ##################################
        # Figure init
        ##################################
        
        fig = plt.figure(figsize= self.figsize, dpi=self.dpi, frameon=True)
        fig.canvas.manager.set_window_title( name )
        ax  = fig.add_subplot(1, 1, 1)

        xname = self.sys.state_label[x] + ' ' + self.sys.state_units[x]
        yname = self.sys.state_label[y] + ' ' + self.sys.state_units[y]
        
        ax.set_ylabel(yname, fontsize=self.fontsize)
        ax.set_xlabel(xname, fontsize=self.fontsize)
        
        x_level = self.x_level[ x ]
        y_level = self.x_level[ y ]
        
        ##################################
        ### Create grid of data and plot
        #################################
        
        #J_grid_nd = np.clip( self.get_grid_from_array( J ) , jmin , jmax )
        J_grid_nd = self.get_grid_from_array( J ) 
        
        J_grid_2d = self.get_2D_slice_of_grid( J_grid_nd , x , y )
        
        mesh = ax.pcolormesh( x_level, y_level, J_grid_2d.T, 
                       shading='gouraud' , cmap = cmap) #, norm = colors.LogNorm()

        mesh.set_clim(vmin=jmin, vmax=jmax)
        
        ##################################
        # Figure param
        ##################################
        
        ax.tick_params( labelsize = self.fontsize )
        ax.grid(True)
        
        fig.colorbar( mesh )
        fig.tight_layout()
        fig.show()
        
        return fig, ax, mesh
                
        
    ##############################
    def plot_control_input_from_policy(self, pi , k , i = 0 , j = 1 ):
        """  
        """
        
        uk_array = self.get_input_from_policy( pi , k )
        
        fig, ax, mesh = self.plot_grid_value( uk_array , self.sys.input_label[ k ] , i , j , self.sys.u_ub[k] , self.sys.u_lb[k] , cmap = 'bwr')
        
        return fig, ax, mesh



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """

    from pyro.dynamic  import pendulum

    sys  = pendulum.SinglePendulum()
    
    G =  GridDynamicSystem( sys )
    
    sys.x_ub = np.array([2.0,2.0])
    sys.x_lb = np.array([-2.0,-2.0])
    sys.u_ub = np.array([1.0,1.0])
    sys.u_lb = np.array([0.0,0.0])
    
    g = GridDynamicSystem( sys , [ 5, 5] , [2] )
    
    
