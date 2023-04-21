#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 05:49:06 2021

@author: alex
"""

import numpy as np
from scipy.optimize import minimize

from pyro.analysis import simulation



###############################################################################
class DirectCollocationTrajectoryOptimisation:
    """ 
    Trajectory optimisation based on fixed-time-steps direct collocation
    ---------------------------------------------------------------------
    sys  : dynamical system class
    dt   : time step size
    grid : number of time step (discretization number)
    
    
    """
    
    ############################
    def __init__(self, sys , dt = 0.2 , grid = 20):
        
        self.sys = sys          # Dynamic system class
        
        self.cost_function = sys.cost_function # default is quadratic cost
        
        self.x_start = sys.x0
        self.x_goal  = sys.xbar
        
        self.dt    = dt
        self.grid  = grid
        
        
        #Parameters
        self.EPS     = 0.01
        self.maxiter = 100
        
        self.compute_bounds()
        
        #Init
        self.dec_init   = np.zeros(grid*(sys.n+sys.m))
        self.iter_count = 0
        
    
    ############################
    def decisionvariables2xu(self, dec):
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
    
        x = np.zeros((n,grid)) 
        u = np.zeros((m,grid))
        
        for i in range(self.sys.n):
            x[i,:] = dec[ i * grid : (i+1) * grid ]
            
        for j in range(self.sys.m):
            k = n + j
            u[j,:] = dec[ k * grid : (k+1) * grid ]
        
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
        
        return dec
    
    ############################
    def set_initial_trajectory_guest(self, traj):
        
        new_traj      = traj.re_sample( self.grid )
        self.dec_init = self.traj2decisionvariables( new_traj )
    
    
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
    
        x,u = self.decisionvariables2xu( dec )
        
        residues_vec = np.zeros( (self.grid-1) * self.sys.n )
        
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
            
            diff = delta_x_num - delta_x_eqs
            
            for j in range(self.sys.n):
                #TODO numpy manip for replacing slow for loop
                residues_vec[i + (self.grid-1) * j ] = diff[j]
            
        return residues_vec
    
    
    ##############################
    def compute_bounds(self):
        """ Compute lower and upper bound vector for all decision variables"""
    
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
    def display_callback(self, a ):
        
        self.iter_count = self.iter_count + 1
        
        print('Optimizing trajectory: Iteration', self.iter_count)
        
        
    ##############################
    def compute_optimal_trajectory(self):
        
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
    def show_solution(self):
        
        self.sys.traj = self.traj
        self.sys.plot_trajectory('xu')
        
    ##############################
    def animate_solution(self, **kwargs):
        
        animator = self.sys.get_animator()
        animator.animate_simulation( self.traj, **kwargs)
        
    ##############################
    def animate_solution_to_html(self, **kwargs):
        
        animator = self.sys.get_animator()
        animator.animate_simulation( self.traj, show = False , **kwargs)
        
        return animator.ani.to_html5_video()
    
        
    ##############################
    def save_solution(self, name = 'optimized_trajectory.npy' ):
        
        self.sys.traj.save( name )
    



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
    
    planner.compute_optimal_trajectory()
    planner.show_solution()