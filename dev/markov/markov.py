#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 09:07:29 2022

@author: alex
"""

import numpy as np


###############################################################################
class MarkovSystem:
    """ 
    Mother class for discrete stochastic dynamical systems
    ----------------------------------------------
    n : number of finite states
    m : number of control action
    p : number of outputs
    k : time index
    ---------------------------------------
    x_k+1 = fk( x_k , u_k , k )
    y_k   = hk( x_k , u_k , k )
    
    optionnal: 
    u_k = policy( y_k , k ) : action autonomous decision
    
    
    """
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################
    
    ############################
    def __init__(self, n = 1, m = 1):
        """ 
        The __init__ method of the Mother class can be used to fill-in default
        labels, units, bounds, and base values.
        """
        
        #############################
        # Parameters
        #############################
        
        p = n

        # Dimensions
        self.n = n   
        self.m = m   
        self.p = p
        
        # Labels
        self.name = 'ContinuousDynamicSystem'
        self.state_label  = []
        self.input_label  = []
        self.output_label = []
        
        # Default Label and units
        for i in range(n):
            self.state_label.append('State '+str(i))
        for i in range(m):
            self.input_label.append('Action '+str(i))
        for i in range(p):
            self.output_label.append('Output '+str(i))
        
        # Default state and inputs values    
        self.xbar = 0
        self.ubar = 0
        
        ################################
        # Transition Probabilities
        ################################
        
        self.T_jia = np.zeros((n,n,m))
        
        for a in range(m):
            
            self.T_jia[:,:,a] = np.diag(np.ones((n)))
            
        ################################
        # Transition cost
        ################################
            
        self.a_jia = np.zeros((n,n,m))
        
        ################################
        # Final cost
        ################################
            
        self.gN_i = np.zeros((n))
        
        
        ################################
        # Variables
        ################################
        
        # Initial value for simulations
        self.x0    = np.zeros( n )
        self.x0[0] = 1
        
        # Result of last simulation
        self.traj = None
        
        # Cost function for evaluation
        # TODO

    
    #############################
    def fk( self , x , u , k = 0 ):
        """ 
        compute the evolution of the probability distribution
        
        """
        
        T_ji = self.T_jia[:,:,u] # transition matrix of given action
        
        x_k1 = np.dot( T_ji , x )
        
        return x_k1
    
    
    #############################
    def check_probability_matrix( self ):
        """ 
        check if transition prob sums to 1
        
        """
        
        print( self.T_jia.sum( axis = 0 ) )
        
        return self.T_jia.sum( axis = 0 ) # should be all ones
    
    
    ###########################################################################
    # The following functions can be overloaded when necessary by child classes
    ###########################################################################
    
    #############################
    def h( self , x , k = 0 ):
        """ 
        Output fonction y = h(x,u,t)
        
        """
        
        y = x      # default output is state
        
        return y
    
    #############################
    def policy( self , y , k ):
        """ 
        
        """
        
        # Default action
        u = self.ubar
        
        return u
    
        
    #############################
    def simulation_of_density_probability( self , N = 10 , plot = True ):
        """ 
        N = number of steps
        """
        
        x_k = self.x0
        
        for k in range(N):
            
            y_k  = self.h( x_k , k )
            u_k  = self.policy( y_k , k )
            x_k1 = self.fk( x_k, u_k , k )
            
            x_k = x_k1
            
            if plot:
                print(x_k1)
            
        
        return x_k1 # return probability distribution at k = N
    
    
    #############################
    def get_valueiteration_algo(self):
        
        vi = ValueIterationForMarkovProcess(self.T_jia, self.a_jia, self.gN_i)
        
        return vi
    
    
    
###############################################################################
class ValueIterationForMarkovProcess:
    """ 
    
    
    
    """
    
    ############################
    def __init__(self, T_jia , a_jia, gN_i ):
        
        
        self.alpha = 1.0 # discount factor
        
        
        self.T = T_jia
        self.a = a_jia
        self.g = gN_i
        
        self.n = T_jia.shape[0] # Number of states
        self.m = T_jia.shape[2] # Number of actions
        
        
        # Initialise cost-to-go with final cost
        self.J = self.g.copy()
        
        # Initialise policy map
        self.c = np.zeros((self.n))
        
        # Initialise Q-values
        self.Q = np.zeros((self.n,self.m))
        
        
        self.print = True
        
        
    ###############################
    def compute_backward_step(self):
        
        # For all states
        for i in range(self.n):
            
            # For all actions
            for a in range(self.m):
                
                Q_j    = self.a[:,i,a] + self.alpha * self.J  # array of possible cost 
                
                self.Q[i,a] = np.dot( self.T[:,i,a] , Q_j )  # expected value
                
        
        self.J = self.Q.min(1)     # Minimum over all possible actions
        self.c = self.Q.argmin(1)  # Action that minimise Q for all i
        
        
    ###############################
    def compute_n_backward_steps(self, n):
        
        for k in range(n):
            
            self.compute_backward_step()
        
            if self.print:
                print('Backward step N-',k)
                print('J = ',self.J)
                print('c = ',self.c)
        
        
            
        
        

    

    
    


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    # 
    m = MarkovSystem(5,3)
    
    
    m.check_probability_matrix()
    
    vi = m.get_valueiteration_algo()
    
    vi.alpha = 0.9
    
    vi.compute_n_backward_steps(100)
    