# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:41:26 2018

@author: Alexandre
"""

###############################################################################
from pyro.analysis import Trajectory
from pyro.control  import controller



###############################################################################
class OpenLoopController( controller.StaticController ) :
    """  
    Open-loop controller based on trajectory solution  
    ----------------------------------------------------
    u = c( x , r, ,t ) = u(t)
    
    """
    
    
    ############################
    def __init__(self, trajectory ):
        """ """
        
        # Sys
        self.trajectory = trajectory
        
        # Dimensions
        self.k = 1   
        self.m = trajectory.m
        self.n = trajectory.n

        if trajectory.y is not None:
            self.p = trajectory.y.shape[1]
        else:
            self.p = trajectory.x.shape[1]

        controller.StaticController.__init__( self, self.k, self.m, self.p)
        
        # Label
        self.name = 'Open Loop Controller'

    #############################
    def c( self , y , r , t  ):
        """  U depends only on time """
        
        u = self.trajectory.t2u( t )
        
        return u

    @staticmethod
    def load_from_file(name):
        traj = Trajectory.load(name)
        return OpenLoopController(traj)

    @property
    def time_final(self):
        return self.trajectory.time_final
    
    


###############################################################################
class Planner:
    """  
    Open-loop controller based on trajectory solution  
    ----------------------------------------------------
    u = c( x , r, ,t ) = u(t)
    
    """
    
    
    ############################
    def __init__(self, sys , cost_function = None ):
        """ """
        
        # Dynamic system model and constraints
        self.sys = sys
        
        # Cost function
        if cost_function is None:
            self.cost_function = sys.cost_function # default is quadratic cost
        else:
            self.cost_function = cost_function
        
        # Start and goal state
        self.x_start = sys.x0
        self.x_goal  = sys.xbar
        
        # Output variable
        self.traj = None
        
        
    ##############################
    def compute_solution(self):
        
        ################################################
        # Place holder: override this method in child classes
        raise NotImplementedError
        ################################################
        
        
    ##############################
    def show_solution(self, plot='xu', **kwargs):
        """ Plot computed trajectory solution """
        
        plotter = self.sys.get_plotter()
        
        return plotter.plot( self.traj, plot, **kwargs)
    
        
    ##############################
    def animate_solution(self, **kwargs):
        
        animator = self.sys.get_animator()
        self.ani = animator.animate_simulation( self.traj, **kwargs)
        
        return self.ani
    
        
    ##############################
    def animate_solution_to_html(self, **kwargs):
        
        animator = self.sys.get_animator()
        animator.animate_simulation( self.traj, show = False , **kwargs)
        
        return animator.ani.to_html5_video()
    
        
    ##############################
    def save_solution(self, name = 'planner_trajectory_solution.npy' ):
        
        self.traj.save( name )
        