#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 05:49:06 2021

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt

import time

from scipy.optimize import minimize

from pyro.analysis import simulation
from pyro.planning import plan
from pyro.analysis import graphical


###############################################################################
class DirectCollocationTrajectoryOptimisation( plan.Planner ):
    """ 
    Trajectory optimisation based on fixed-time-steps direct collocation
    ---------------------------------------------------------------------
    sys  : dynamical system class
    dt   : time step size
    grid : number of time step (discretization number)
    cf   : cost function class
    
    
    """
    
    ############################
    def __init__(self, sys , dt = 0.2 , grid = 20 , cost_function = None,  dynamic_plot = False ):
        
        
        # Set sys, default cost function x_start and x_goal
        plan.Planner.__init__(self, sys , cost_function )
        
        # Discretization parameters
        self.dt    = dt
        self.grid  = grid
        
        # Parameters
        self.EPS     = 0.01
        self.maxiter = 100
        
        # Initial computation
        self.compute_bounds()
        self.dec_init   = np.zeros( grid * ( sys.n + sys.m ) )
        
        
        # Check if vectorized operation are available
        try:
            is_vectorized = self.sys.is_vectorized & self.cost_function.is_vectorized
        except:
            is_vectorized = False
        self.is_vectorized = is_vectorized
            
        
        
        # Memory variable
        self.iter_count = 0
        self.start_time = time.time()
        
        # Optional Convergence Graph
        self.dynamic_plot = dynamic_plot
        
        if dynamic_plot:
            
            self.init_dynamic_plot()
            
            
    ############################
    def init_dynamic_plot(self):
        
        # Graphic option
        self.dynamic_plot = True
            
        traj = self.decisionvariables2traj( self.dec_init )
        self.plotter = graphical.TrajectoryPlotter( self.sys )
        self.plotter.plot( traj, 'xu' )
        
        
    ############################
    def set_initial_trajectory_guest(self, traj):
        
        new_traj      = traj.re_sample( self.grid )
        self.dec_init = self.traj2decisionvariables( new_traj )
        
        if self.dynamic_plot:
            
            traj = self.decisionvariables2traj( self.dec_init )
            self.plotter.update_plot( traj, 'xu' )
            plt.pause( 0.001 )
            
    ############################
    def set_linear_initial_guest(self, derivatives = False ):
        
        xs = np.linspace(  self.x_start,   self.x_goal, self.grid )
        us = np.linspace( self.sys.ubar, self.sys.ubar, self.grid )
        
        # For second order mechanical system
        if derivatives:
            dof = int(self.sys.n/2)
            tf  = self.grid * self.dt
            dx  = ( self.x_goal[:dof] - self.x_start[:dof] ) / tf
            dxs = np.linspace( dx, dx, self.grid )
            xs[:,dof:] = dxs
        
        
        dec = np.array([]).reshape(0,1) # initialize dec_vars array
        
        for i in range(self.sys.n): # append states x
            arr_to_add = xs[:,i].reshape(self.grid,1)
            dec = np.append(dec,arr_to_add,axis=0)
    
        for i in range(self.sys.m): # append inputs u
            arr_to_add = us[:,i].reshape(self.grid,1)
            dec = np.append(dec,arr_to_add,axis=0)
            
        self.dec_init = dec[:,0]
        
        
        if self.dynamic_plot:
            
            traj = self.decisionvariables2traj( self.dec_init )
            self.plotter.update_plot( traj, 'xu' )
            plt.pause( 0.001 )
        
    
    ############################
    def decisionvariables2xu(self, dec ):
        """ 
        Unpack decision variable vector into x and u trajectory matrices 
        --------------------------
        
        dec = [ x[0](t=0), .... x[0](t), .... x[0](t=tf), 
               ...
               x[i](t=0), .... x[i](t), .... x[i](t=tf), 
               ...
               x[n](t=0), .... x[n](t), .... x[n](t=tf), 
               
               u[0](t=0), .... u[0](t), .... u[0](t=tf), 
               ...
               u[j](t=0), .... u[j](t), .... u[j](t=tf), 
               ...
               u[m](t=0), .... u[m](t), .... u[m](t=tf) ]
        
        """
        
        n    = self.sys.n   # number of states
        m    = self.sys.m   # number of inputs
        grid = self.grid    # number of time steps
    
        # x = np.zeros( ( n , grid ) ) 
        # u = np.zeros( ( m , grid ) )
        
        # # Extract states variables
        # for i in range(self.sys.n):
        #     x[i,:] = dec[ i * grid : (i+1) * grid ]
            
        # # Extract input variables
        # for j in range(self.sys.m):
        #     k = n + j
        #     u[j,:] = dec[ k * grid : (k+1) * grid ]
        
        x = dec[: n * grid ].reshape( n , grid )
        u = dec[ n * grid :].reshape( m , grid )
        
        return x,u
    
    
    ############################
    def traj2decisionvariables(self, traj):
        """ 
        Compute decision variables based onna trajectory object
        --------------------------
        
        dec = [ x[0](t=0), .... x[0](t), .... x[0](t=tf), 
               ...
               x[i](t=0), .... x[i](t), .... x[i](t=tf), 
               ...
               x[n](t=0), .... x[n](t), .... x[n](t=tf), 
               
               u[0](t=0), .... u[0](t), .... u[0](t=tf), 
               ...
               u[j](t=0), .... u[j](t), .... u[j](t=tf), 
               ...
               u[m](t=0), .... u[m](t), .... u[m](t=tf) ]
        
        """
        
        #n = grid*(self.sys.n+self.sys.m)
        
        dec = np.array([]).reshape(0,1) # initialize dec_vars array
        
        for i in range(self.sys.n): # append states x
            arr_to_add = traj.x[:,i].reshape(self.grid,1)
            dec = np.append(dec,arr_to_add,axis=0)
    
        for i in range(self.sys.m): # append inputs u
            arr_to_add = traj.u[:,i].reshape(self.grid,1)
            dec = np.append(dec,arr_to_add,axis=0)
        
        return dec[:,0]
    
    
    ############################
    def decisionvariables2traj(self, dec):
        """ 
        Unpack decision variable vector into x and u trajectory matrices 
        --------------------------
        
        dec = [ x[0](t=0), .... x[0](t), .... x[0](t=tf), 
               ...
               x[i](t=0), .... x[i](t), .... x[i](t=tf), 
               ...
               x[n](t=0), .... x[n](t), .... x[n](t=tf), 
               
               u[0](t=0), .... u[0](t), .... u[0](t=tf), 
               ...
               u[j](t=0), .... u[j](t), .... u[j](t=tf), 
               ...
               u[m](t=0), .... u[m](t), .... u[m](t=tf) ]
        
        """
        
        n    = self.sys.n   # number of states
        m    = self.sys.m   # number of inputs
        p    = self.sys.p   # number of inputs
        grid = self.grid    # number of time steps
    
        x = np.zeros((grid,n)) 
        u = np.zeros((grid,m))
        t  = np.zeros(grid)
        y  = np.zeros(( grid, p ))
        dx = np.zeros(( grid, n ))
        dJ = np.zeros(grid)
        J  = np.zeros(grid)
        
        J_sum = 0
        
        xT,uT = self.decisionvariables2xu( dec )
        
        for i in range(self.grid):
            x[i,:]  = xT[:,i]
            u[i,:]  = uT[:,i]
            t[i]    = i*self.dt
            dx[i,:] = self.sys.f(x[i,:],u[i,:],t[i])
            dJ[i]   = self.cost_function.g(x[i,:],u[i,:],t[i])
            
            J_sum = J_sum + dJ[i]
            J[i]  = J_sum
            
        #########################
        traj = simulation.Trajectory(
          x = x,
          u = u,
          t = t,
          dx= dx,
          y = y,
          dJ = dJ,
          J  = J
        )
        
        self.traj = traj
        
        return traj
        
    
    ############################
    def cost(self, dec):
        """ Compute cost for given decision variable using trapez integration approx """
        
        x,u = self.decisionvariables2xu( dec )
        
        if self.is_vectorized:
            
            # Vectorized operation version
            t  = np.linspace(0, ( self.grid - 1 )* self.dt, self.grid)
            
            dJ = self.cost_function.g( x , u , t )
            
            J  = np.trapz( dJ , t )
            
        else:
            
            # Loop version
        
            J = 0
            
            for i in range(self.grid -1):
                #i
                x_i = x[:,i]
                u_i = u[:,i]
                t_i = i*self.dt
                dJi = self.cost_function.g( x_i , u_i, t_i )
                
                #i+1
                x_i1 = x[:,i+1]
                u_i1 = u[:,i+1]
                t_i1 = (i+1)*self.dt
                dJi1 = self.cost_function.g( x_i1 , u_i1, t_i1 )
                
                #trapez
                dJ = 0.5 * ( dJi + dJi1 )
                
                #integral
                J = J + dJ * self.dt
            
        return J
    
    
    ########################
    def dynamic_constraints(self, dec):
        """ Compute residues of dynamic constraints """
    
        x , u = self.decisionvariables2xu( dec )
        
        if self.is_vectorized:
            
            # Vectorized operation version
            
            t  = np.linspace(0, ( self.grid - 1 )* self.dt, self.grid)
            
            dx = self.sys.f( x ,u , t )
            
            dx_eqs = 0.5 * ( dx[:,0:-1] + dx[:,1:] ) * self.dt
            
            dx_num = np.diff( x )
            
            residues = dx_num - dx_eqs
            
        
        else:
            
            # Loop version
        
            residues = np.zeros( ( self.grid - 1 , self.sys.n  ))
            
            for i in range(self.grid-1):
                
                #i
                x_i = x[:,i]
                u_i = u[:,i]
                t_i = i*self.dt
                dx_i = self.sys.f(x_i,u_i,t_i) # analytical state derivatives
                
                #i+1
                x_i1 = x[:,i+1]
                u_i1 = u[:,i+1]
                t_i1 = (i+1)*self.dt
                dx_i1 = self.sys.f(x_i1,u_i1,t_i1) # analytical state derivatives
                
                #trapez
                delta_x_eqs = 0.5 * self.dt * (dx_i + dx_i1)
                
                #num diff
                delta_x_num = x[:,i+1] - x[:,i] # numerical delta in trajectory data
                
                residues[i,:] = delta_x_num - delta_x_eqs
            
        return residues.flatten()
    
    
    ##############################
    def compute_bounds(self):
        """ Compute lower and upper bound vector for all decision variables """
    
        bounds = []
        
        # States constraints
        for j in range(self.sys.n):
            
            # initial states
            bounds.append( ( self.x_start[j] , self.x_start[j] + self.EPS ) )
            
            # range for intermediate states
            for i in range(1,self.grid - 1 ):
                bounds.append( ( self.sys.x_lb[j] , self.sys.x_ub[j] ) )
                
            # final goal state
            bounds.append( ( self.x_goal[j] , self.x_goal[j] + self.EPS ) )
        
        
        # Ipnut constraints
        for j in range(self.sys.m):

            for i in range(0,self.grid):
                bounds.append( ( self.sys.u_lb[j] , self.sys.u_ub[j] ) )
        
            
        self.bounds = bounds
    
    
    ##############################
    def display_callback(self, x ):
        
        self.iter_count = self.iter_count + 1
        
        print('Optimizing trajectory: iteration no', self.iter_count , 
              ' elapsed time = %.2f' % (time.time() - self.start_time ) )
        
        if self.dynamic_plot:
            
            traj = self.decisionvariables2traj( x )
            self.plotter.update_plot( traj, 'xu' )
            plt.pause( 0.001 )
        
        
        
    ##############################
    def compute_optimal_trajectory(self):
        
        self.start_time = time.time()
        
        self.compute_bounds()
        
        dynamic_constraints = {'type': 'eq', 'fun': self.dynamic_constraints }
    
        res = minimize(self.cost, 
                       self.dec_init, 
                       method='SLSQP',  
                       bounds=self.bounds, 
                       constraints=dynamic_constraints, 
                       callback=self.display_callback, 
                       options={'disp':True,'maxiter':self.maxiter})
        
        self.res  = res
        self.traj = self.decisionvariables2traj( self.res.x )
        
        
    ##############################
    def compute_solution(self):
        
        self.compute_optimal_trajectory()
        
        return self.traj
    


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    from pyro.dynamic import pendulum
    

    sys  = pendulum.SinglePendulum()
    
    planner = DirectCollocationTrajectoryOptimisation( sys )
    
    planner.x_start = np.array([0.1,0])
    planner.x_goal  = np.array([-3.14,0])
    
    planner.init_dynamic_plot()
    
    planner.compute_solution()
    planner.animate_solution()

    
    