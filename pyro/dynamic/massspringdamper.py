# -*- coding: utf-8 -*-
"""
Created on 4 Jun 2021

@author: Alexandre
"""

import numpy as np


from pyro.dynamic import statespace


###############################################################################

class SingleMass( statespace.StateSpaceSystem ):
    """Single Mass with linear spring and damper

    Attributes
    ----------

    """

    ############################
    def __init__(self, m=1, k=2, b=0):
        """ """

        # params
        self.m = m
        self.k = k
        self.b = b
        
        self.l1 = 2
        self.l2 = 1
        
        # Matrix ABCD
        self.compute_ABCD()
        
        # initialize standard params
        super().__init__(self.A, self.B, self.C, self.D)
        
        # Name and labels
        self.name = 'Linear-Spring-Damper'
        self.input_label = [ 'Force']
        self.input_units = [ '[N]']
        self.output_label = ['Position']
        self.output_units = ['[m]']
        self.state_label = [ 'Position','Velocity']
        self.state_units = [ '[m]', '[m/s]']
        
        self.linestyle = '-'
        
    
    ###########################################################################
    def compute_ABCD(self):
        """ 
        """
        self.A = np.array([ [ 0              ,              1 ], 
                            [ -self.k/self.m , -self.b/self.m ] ])
        self.B = np.array([ [ 0 ],
                            [ 1 /self.m ]])
        self.C = np.array([ [ 1 , 0 ]])
        self.D = np.array([ [ 0 ]])
                
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    #############################
    def xut2q( self, x , u , t ):
        """ Compute configuration variables ( q vector ) """
        
        q = np.array([ x[0], u[0] ]) # Hack to illustrate force vector

        return q
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.l1 * 2
        
        domain  = [ (-l+self.l1,l+self.l1) , (-l,l) , (-l,l) ]#  
                
        return domain
    
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts   = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        # ground line
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-self.l1,-self.l2,0])
        pts[1,:] = np.array([-self.l1,+self.l2,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
        
        # mass
        pts      = np.zeros(( 5 , 3 ))
        
        pts[0,:] =  np.array([q[0] - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([q[0] + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([q[0] + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([q[0] - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        # spring
        pts      = np.zeros(( 15 , 3 ))
        
        d = q[0] + self.l1 - self.l2/2
        h =  self.l2 / 3
        
        pts[0,:]  = np.array([d*0.00 - self.l1,0,0])
        pts[1,:]  = np.array([d*0.20 - self.l1,0,0])
        pts[2,:]  = np.array([d*0.25 - self.l1,+h,0])
        pts[3,:]  = np.array([d*0.30 - self.l1,-h,0])
        pts[4,:]  = np.array([d*0.35 - self.l1,+h,0])
        pts[5,:]  = np.array([d*0.40 - self.l1,-h,0])
        pts[6,:]  = np.array([d*0.45 - self.l1,+h,0])
        pts[7,:]  = np.array([d*0.50 - self.l1,-h,0])
        pts[8,:]  = np.array([d*0.55 - self.l1,+h,0])
        pts[9,:]  = np.array([d*0.60 - self.l1,-h,0])
        pts[10,:] = np.array([d*0.65 - self.l1,+h,0])
        pts[11,:] = np.array([d*0.70 - self.l1,-h,0])
        pts[12,:] = np.array([d*0.75 - self.l1,+h,0])
        pts[13,:] = np.array([d*0.80 - self.l1,0,0])
        pts[14,:] = np.array([d*1.00 - self.l1,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
                
        return lines_pts , lines_style , lines_color
    
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        plots the force vector
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        # force arrow
        pts      = np.zeros(( 5 , 3 ))
        
        xf = x[0] # base of force x coordinate
        f  = u[0] # force amplitude
        
        pts[0,:] =  np.array([xf + self.l2/2,0,0])
        pts[1,:] =  np.array([xf + self.l2/2 + f,0,0])
        pts[2,:] =  np.array([xf + self.l2/2 + f - self.l2/4*f,+self.l2/4*f,0])
        pts[3,:] =  np.array([xf + self.l2/2 + f,0,0])
        pts[4,:] =  np.array([xf + self.l2/2 + f - self.l2/4*f,-self.l2/4*f,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'r')
                
        return lines_pts , lines_style , lines_color
    


###############################################################################

class TwoMass( statespace.StateSpaceSystem ):
    """Two Mass with linear spring and damper

    Attributes
    ----------

    """

    ############################
    def __init__(self, m=1, k=2, b=0.2, output_mass = 2):
        """ """

        # params
        self.m1 = m
        self.k1 = k
        self.b1 = b
        self.m2 = m
        self.k2 = k
        self.b2 = b
        
        self.l1 = 2
        self.l2 = 1
        
        # sensor output
        self.output_mass = output_mass
        
        # Matrix ABCD
        self.compute_ABCD()
        
        # initialize standard params
        super().__init__(self.A, self.B, self.C, self.D)
        
        # Name and labels
        self.name = 'Two mass with linear spring-dampers'
        self.input_label  = ['Force']
        self.input_units  = ['[N]']
        self.output_label = ['x2']
        self.output_units = ['[m]']
        self.state_label  = [ 'x1','x2', 'dx1', 'dx2']
        self.state_units  = [ '[m]', '[m]', '[m/s]', '[m/s]']
        
        self.linestyle = '-'
        
    
    ###########################################################################
    def compute_ABCD(self):
        """ 
        """
        self.A = np.array([ [ 0, 0, 1, 0 ],
                            [ 0, 0, 0, 1 ],
                            [ -(self.k1+self.k2)/self.m1, +self.k2/self.m1, -self.b1/self.m1, 0],
                            [           +self.k2/self.m2, -self.k2/self.m2, 0, -self.b2/self.m2]])
        self.B = np.array([ [ 0 ],
                            [ 0 ],
                            [ 0 ],
                            [ 1/self.m2 ]])
        
        if self.output_mass == 2:
            self.C = np.array([ [ 0 , 1 , 0 , 0 ]])
            self.output_label = ['x2']
        elif self.output_mass ==1:
            self.C = np.array([ [ 1 , 0 , 0 , 0 ]])
            self.output_label = ['x1']
        else:
            self.C = np.array([ [ 0 , 1 , 0 , 0 ]])
            self.output_label = ['x2']
            
        self.D = np.array([ [ 0 ]])
                
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    #############################
    def xut2q( self, x , u , t ):
        """ Compute configuration variables ( q vector ) """
        
        q = np.array([ x[0], x[1], u[0] ])

        return q
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.l1 * 3
        
        domain  = [ (-l+self.l1,l+self.l1) , (-l,l) , (-l,l) ]#  
                
        return domain
    
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts   = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        # ground line
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-self.l1*2,-self.l2,0])
        pts[1,:] = np.array([-self.l1*2,+self.l2,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
        
        # mass 1 
        pts      = np.zeros(( 5 , 3 ))
        
        x1 = q[0] - self.l1
        
        pts[0,:] =  np.array([ x1 - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([ x1 + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([ x1 + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([ x1 - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'g')
        
        # mass 2 
        pts      = np.zeros(( 5 , 3 ))
        
        x2 = q[1]
        
        pts[0,:] =  np.array([x2 - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([x2 + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([x2 + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([x2 - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        # spring 1 
        pts      = np.zeros(( 15 , 3 ))
        
        d = q[0] + self.l1 - self.l2/2
        h =  self.l2 / 3
        
        pts[0,:]  = np.array([d*0.00 - self.l1*2,0,0])
        pts[1,:]  = np.array([d*0.20 - self.l1*2,0,0])
        pts[2,:]  = np.array([d*0.25 - self.l1*2,+h,0])
        pts[3,:]  = np.array([d*0.30 - self.l1*2,-h,0])
        pts[4,:]  = np.array([d*0.35 - self.l1*2,+h,0])
        pts[5,:]  = np.array([d*0.40 - self.l1*2,-h,0])
        pts[6,:]  = np.array([d*0.45 - self.l1*2,+h,0])
        pts[7,:]  = np.array([d*0.50 - self.l1*2,-h,0])
        pts[8,:]  = np.array([d*0.55 - self.l1*2,+h,0])
        pts[9,:]  = np.array([d*0.60 - self.l1*2,-h,0])
        pts[10,:] = np.array([d*0.65 - self.l1*2,+h,0])
        pts[11,:] = np.array([d*0.70 - self.l1*2,-h,0])
        pts[12,:] = np.array([d*0.75 - self.l1*2,+h,0])
        pts[13,:] = np.array([d*0.80 - self.l1*2,0,0])
        pts[14,:] = np.array([d*1.00 - self.l1*2,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
        
        # spring 2 
        pts      = np.zeros(( 15 , 3 ))
        
        d = q[1] - q[0] + self.l1 - self.l2
        h =  self.l2 / 3
        
        pts[0,:]  = np.array([d*0.00 + x1 + self.l2/2,0,0])
        pts[1,:]  = np.array([d*0.20 + x1+self.l2/2,0,0])
        pts[2,:]  = np.array([d*0.25 + x1+self.l2/2,+h,0])
        pts[3,:]  = np.array([d*0.30 + x1+self.l2/2,-h,0])
        pts[4,:]  = np.array([d*0.35 + x1+self.l2/2,+h,0])
        pts[5,:]  = np.array([d*0.40 + x1+self.l2/2,-h,0])
        pts[6,:]  = np.array([d*0.45 + x1+self.l2/2,+h,0])
        pts[7,:]  = np.array([d*0.50 + x1+self.l2/2,-h,0])
        pts[8,:]  = np.array([d*0.55 + x1+self.l2/2,+h,0])
        pts[9,:]  = np.array([d*0.60 + x1+self.l2/2,-h,0])
        pts[10,:] = np.array([d*0.65 + x1+self.l2/2,+h,0])
        pts[11,:] = np.array([d*0.70 + x1+self.l2/2,-h,0])
        pts[12,:] = np.array([d*0.75 + x1+self.l2/2,+h,0])
        pts[13,:] = np.array([d*0.80 + x1+self.l2/2,0,0])
        pts[14,:] = np.array([d*1.00 + x1+self.l2/2,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
                
        return lines_pts , lines_style , lines_color
    
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        plots the force vector
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        # force arrow
        pts      = np.zeros(( 5 , 3 ))
        
        xf = x[1] # base of force x coordinate
        f  = u[0] # force amplitude
        
        pts[0,:] =  np.array([xf + self.l2/2,0,0])
        pts[1,:] =  np.array([xf + self.l2/2 + f,0,0])
        pts[2,:] =  np.array([xf + self.l2/2 + f - self.l2/4*f,+self.l2/4*f,0])
        pts[3,:] =  np.array([xf + self.l2/2 + f,0,0])
        pts[4,:] =  np.array([xf + self.l2/2 + f - self.l2/4*f,-self.l2/4*f,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'r')
                
        return lines_pts , lines_style , lines_color
  
    
  ###############################################################################
    
    
class ThreeMass( statespace.StateSpaceSystem ):
    """Three Mass with linear spring and damper

    Attributes
    ----------

    """

    ############################
    def __init__(self, m=1, k=2, b=0.2, output_mass = 2):
        """ """

        # params
        self.m1 = m
        self.k1 = k
        self.b1 = b
        self.m2 = m
        self.k2 = k
        self.b2 = b
        self.m3 = m
        self.k3 = k
        self.b3 = b
        
        self.l1 = 2
        self.l2 = 1
        
        # sensor output
        self.output_mass = output_mass
        
        # Matrix ABCD
        self.compute_ABCD()
        
        # initialize standard params
        super().__init__(self.A, self.B, self.C, self.D)
        
        # Name and labels
        self.name = 'Three mass with linear spring-dampers'
        self.input_label  = ['Force']
        self.input_units  = ['[N]']
        self.output_label = ['x3']
        self.output_units = ['[m]']
        self.state_label  = [ 'x1','x2', 'x3', 'dx1', 'dx2', 'dx3']
        self.state_units  = [ '[m]', '[m]', '[m]', '[m/s]', '[m/s]','[m/s]']
        
        self.linestyle = '-'
        
    
    ###########################################################################
    def compute_ABCD(self):
        """ 
        """
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        m1 = self.m1
        m2 = self.m2
        m3 = self.m3
        b1 = self.b1
        b2 = self.b2
        b3 = self.b3
        
        self.A = np.array([ [ 0, 0, 0, 1, 0, 0 ],
                            [ 0, 0, 0, 0, 1, 0 ],
                            [ 0, 0, 0, 0, 0, 1 ],
                            [ -(k1+k2)/m1,      +k2/m1,      0, -b1/m1,      0,      0 ],
                            [      +k2/m2, -(k2+k3)/m2, +k3/m2,      0, -b2/m2,      0 ],
                            [           0,      +k3/m3, -k3/m3,      0,      0, -b3/m3 ]])
        self.B = np.array([ [ 0 ],
                            [ 0 ],
                            [ 0 ],
                            [ 0 ],
                            [ 0 ],
                            [ 1/m2 ]])
        
        if self.output_mass == 3:
            self.C = np.array([ [ 0 , 0 , 1 , 0 , 0 , 0 ]])
            self.output_label = ['x3']
        elif self.output_mass == 2:
            self.C = np.array([ [ 0 , 1 , 0 , 0 , 0 , 0 ]])
            self.output_label = ['x2']
        elif self.output_mass == 1:
            self.C = np.array([ [ 1 , 0 , 0 , 0 , 0 , 0 ]])
            self.output_label = ['x1']
        else:
            self.C = np.array([ [ 0 , 0 , 1 , 0 , 0 , 0 ]])
            self.output_label = ['x3']
            
        self.D = np.array([ [ 0 ]])
                
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    #############################
    def xut2q( self, x , u , t ):
        """ Compute configuration variables ( q vector ) """
        
        q = np.array([ x[0], x[1], x[2], u[0] ])

        return q
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.l1 * 3
        
        domain  = [ (-l+self.l1,l+self.l1) , (-l,l) , (-l,l) ]#  
                
        return domain
    
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts   = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        # ground line
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-self.l1*2,-self.l2,0])
        pts[1,:] = np.array([-self.l1*2,+self.l2,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
        
        # mass 1 
        pts      = np.zeros(( 5 , 3 ))
        
        x1 = q[0] - self.l1
        
        pts[0,:] =  np.array([ x1 - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([ x1 + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([ x1 + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([ x1 - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'm')
        
        # mass 2 
        pts      = np.zeros(( 5 , 3 ))
        
        x2 = q[1]
        
        pts[0,:] =  np.array([x2 - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([x2 + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([x2 + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([x2 - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'g')
        
        #mass 3
        pts      = np.zeros(( 5 , 3 ))
        
        x3 = q[2] + self.l1
        
        pts[0,:] =  np.array([x3 - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([x3 + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([x3 + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([x3 - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        # spring 1 
        pts      = np.zeros(( 15 , 3 ))
        
        d = q[0] + self.l1 - self.l2/2
        h =  self.l2 / 3
        
        pts[0,:]  = np.array([d*0.00 - self.l1*2,0,0])
        pts[1,:]  = np.array([d*0.20 - self.l1*2,0,0])
        pts[2,:]  = np.array([d*0.25 - self.l1*2,+h,0])
        pts[3,:]  = np.array([d*0.30 - self.l1*2,-h,0])
        pts[4,:]  = np.array([d*0.35 - self.l1*2,+h,0])
        pts[5,:]  = np.array([d*0.40 - self.l1*2,-h,0])
        pts[6,:]  = np.array([d*0.45 - self.l1*2,+h,0])
        pts[7,:]  = np.array([d*0.50 - self.l1*2,-h,0])
        pts[8,:]  = np.array([d*0.55 - self.l1*2,+h,0])
        pts[9,:]  = np.array([d*0.60 - self.l1*2,-h,0])
        pts[10,:] = np.array([d*0.65 - self.l1*2,+h,0])
        pts[11,:] = np.array([d*0.70 - self.l1*2,-h,0])
        pts[12,:] = np.array([d*0.75 - self.l1*2,+h,0])
        pts[13,:] = np.array([d*0.80 - self.l1*2,0,0])
        pts[14,:] = np.array([d*1.00 - self.l1*2,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
        
        # spring 2 
        pts      = np.zeros(( 15 , 3 ))
        
        d = q[1] - q[0] + self.l1 - self.l2
        
        pts[0,:]  = np.array([d*0.00 + x1 + self.l2/2,0,0])
        pts[1,:]  = np.array([d*0.20 + x1+self.l2/2,0,0])
        pts[2,:]  = np.array([d*0.25 + x1+self.l2/2,+h,0])
        pts[3,:]  = np.array([d*0.30 + x1+self.l2/2,-h,0])
        pts[4,:]  = np.array([d*0.35 + x1+self.l2/2,+h,0])
        pts[5,:]  = np.array([d*0.40 + x1+self.l2/2,-h,0])
        pts[6,:]  = np.array([d*0.45 + x1+self.l2/2,+h,0])
        pts[7,:]  = np.array([d*0.50 + x1+self.l2/2,-h,0])
        pts[8,:]  = np.array([d*0.55 + x1+self.l2/2,+h,0])
        pts[9,:]  = np.array([d*0.60 + x1+self.l2/2,-h,0])
        pts[10,:] = np.array([d*0.65 + x1+self.l2/2,+h,0])
        pts[11,:] = np.array([d*0.70 + x1+self.l2/2,-h,0])
        pts[12,:] = np.array([d*0.75 + x1+self.l2/2,+h,0])
        pts[13,:] = np.array([d*0.80 + x1+self.l2/2,0,0])
        pts[14,:] = np.array([d*1.00 + x1+self.l2/2,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
        
        # spring 3
        pts      = np.zeros(( 15 , 3 ))
        
        d = q[2] - q[1] + self.l1 - self.l2
        
        pts[0,:]  = np.array([d*0.00 + x2 + self.l2/2,0,0])
        pts[1,:]  = np.array([d*0.20 + x2+self.l2/2,0,0])
        pts[2,:]  = np.array([d*0.25 + x2+self.l2/2,+h,0])
        pts[3,:]  = np.array([d*0.30 + x2+self.l2/2,-h,0])
        pts[4,:]  = np.array([d*0.35 + x2+self.l2/2,+h,0])
        pts[5,:]  = np.array([d*0.40 + x2+self.l2/2,-h,0])
        pts[6,:]  = np.array([d*0.45 + x2+self.l2/2,+h,0])
        pts[7,:]  = np.array([d*0.50 + x2+self.l2/2,-h,0])
        pts[8,:]  = np.array([d*0.55 + x2+self.l2/2,+h,0])
        pts[9,:]  = np.array([d*0.60 + x2+self.l2/2,-h,0])
        pts[10,:] = np.array([d*0.65 + x2+self.l2/2,+h,0])
        pts[11,:] = np.array([d*0.70 + x2+self.l2/2,-h,0])
        pts[12,:] = np.array([d*0.75 + x2+self.l2/2,+h,0])
        pts[13,:] = np.array([d*0.80 + x2+self.l2/2,0,0])
        pts[14,:] = np.array([d*1.00 + x2+self.l2/2,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
                
        return lines_pts , lines_style , lines_color
    
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        plots the force vector
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        # force arrow
        pts      = np.zeros(( 5 , 3 ))
        
        xf = x[2] + self.l1
        f = u[0]
        
        pts[0,:] =  np.array([xf + self.l2/2,0,0])
        pts[1,:] =  np.array([xf + self.l2/2 + f,0,0])
        pts[2,:] =  np.array([xf + self.l2/2 + f - self.l2/4*f,+self.l2/4*f,0])
        pts[3,:] =  np.array([xf + self.l2/2 + f,0,0])
        pts[4,:] =  np.array([xf + self.l2/2 + f - self.l2/4*f,-self.l2/4*f,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'r')
                
        return lines_pts , lines_style , lines_color
    
    

###############################################################################

class FloatingSingleMass( SingleMass ):
    """Single Mass with damping

    Attributes
    ----------

    """

    ############################
    def __init__(self, m=1, b=0):
        """ """
        super().__init__(m,0,b)
        
        # Name and labels
        self.name = 'Mass'
        
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts   = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []

        # mass
        pts      = np.zeros(( 5 , 3 ))
        
        pts[0,:] =  np.array([q[0] - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([q[0] + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([q[0] + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([q[0] - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
                
        return lines_pts , lines_style , lines_color


###############################################################################

class FloatingTwoMass( TwoMass ):
    """Single Mass with damping

    Attributes
    ----------

    """

    ############################
    def __init__(self, m=1, k=1, b=0, output_mass=2):
        """ """
        super().__init__(m,k,b,output_mass)
        
        self.k1 = 0  # no base spring
        self.compute_ABCD()
        
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        # mass 1 
        pts      = np.zeros(( 5 , 3 ))
        
        x1 = q[0] - self.l1
        
        pts[0,:] =  np.array([ x1 - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([ x1 + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([ x1 + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([ x1 - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'g')
        
        # mass 2 
        pts      = np.zeros(( 5 , 3 ))
        
        x2 = q[1]
        
        pts[0,:] =  np.array([x2 - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([x2 + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([x2 + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([x2 - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        
        # spring 2 
        pts      = np.zeros(( 15 , 3 ))
        
        d = q[1] - q[0] + self.l1 - self.l2
        h =  self.l2 / 3
        
        pts[0,:]  = np.array([d*0.00 + x1 + self.l2/2,0,0])
        pts[1,:]  = np.array([d*0.20 + x1+self.l2/2,0,0])
        pts[2,:]  = np.array([d*0.25 + x1+self.l2/2,+h,0])
        pts[3,:]  = np.array([d*0.30 + x1+self.l2/2,-h,0])
        pts[4,:]  = np.array([d*0.35 + x1+self.l2/2,+h,0])
        pts[5,:]  = np.array([d*0.40 + x1+self.l2/2,-h,0])
        pts[6,:]  = np.array([d*0.45 + x1+self.l2/2,+h,0])
        pts[7,:]  = np.array([d*0.50 + x1+self.l2/2,-h,0])
        pts[8,:]  = np.array([d*0.55 + x1+self.l2/2,+h,0])
        pts[9,:]  = np.array([d*0.60 + x1+self.l2/2,-h,0])
        pts[10,:] = np.array([d*0.65 + x1+self.l2/2,+h,0])
        pts[11,:] = np.array([d*0.70 + x1+self.l2/2,-h,0])
        pts[12,:] = np.array([d*0.75 + x1+self.l2/2,+h,0])
        pts[13,:] = np.array([d*0.80 + x1+self.l2/2,0,0])
        pts[14,:] = np.array([d*1.00 + x1+self.l2/2,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
        
        return lines_pts , lines_style , lines_color
    
    
###############################################################################

class FloatingThreeMass( ThreeMass ):
    """Single Mass with damping

    Attributes
    ----------

    """

    ############################
    def __init__(self, m=1, k=1, b=0, output_mass=3):
        """ """
        super().__init__(m,k,b,output_mass)
        self.k1 = 0
        self.compute_ABCD()
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []

        
        # mass 1 
        pts      = np.zeros(( 5 , 3 ))
        
        x1 = q[0] - self.l1
        
        pts[0,:] =  np.array([ x1 - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([ x1 + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([ x1 + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([ x1 - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'm')
        
        # mass 2 
        pts      = np.zeros(( 5 , 3 ))
        
        x2 = q[1]
        
        pts[0,:] =  np.array([x2 - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([x2 + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([x2 + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([x2 - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'g')
        
        #mass 3
        pts      = np.zeros(( 5 , 3 ))
        
        x3 = q[2] + self.l1
        
        pts[0,:] =  np.array([x3 - self.l2/2,+self.l2/2,0])
        pts[1,:] =  np.array([x3 + self.l2/2,+self.l2/2,0])
        pts[2,:] =  np.array([x3 + self.l2/2,-self.l2/2,0])
        pts[3,:] =  np.array([x3 - self.l2/2,-self.l2/2,0])
        pts[4,:] =  pts[0,:]
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        # spring 2 
        pts      = np.zeros(( 15 , 3 ))
        h =  self.l2 / 3
        d = q[1] - q[0] + self.l1 - self.l2
        
        pts[0,:]  = np.array([d*0.00 + x1 + self.l2/2,0,0])
        pts[1,:]  = np.array([d*0.20 + x1+self.l2/2,0,0])
        pts[2,:]  = np.array([d*0.25 + x1+self.l2/2,+h,0])
        pts[3,:]  = np.array([d*0.30 + x1+self.l2/2,-h,0])
        pts[4,:]  = np.array([d*0.35 + x1+self.l2/2,+h,0])
        pts[5,:]  = np.array([d*0.40 + x1+self.l2/2,-h,0])
        pts[6,:]  = np.array([d*0.45 + x1+self.l2/2,+h,0])
        pts[7,:]  = np.array([d*0.50 + x1+self.l2/2,-h,0])
        pts[8,:]  = np.array([d*0.55 + x1+self.l2/2,+h,0])
        pts[9,:]  = np.array([d*0.60 + x1+self.l2/2,-h,0])
        pts[10,:] = np.array([d*0.65 + x1+self.l2/2,+h,0])
        pts[11,:] = np.array([d*0.70 + x1+self.l2/2,-h,0])
        pts[12,:] = np.array([d*0.75 + x1+self.l2/2,+h,0])
        pts[13,:] = np.array([d*0.80 + x1+self.l2/2,0,0])
        pts[14,:] = np.array([d*1.00 + x1+self.l2/2,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
        
        # spring 3
        pts      = np.zeros(( 15 , 3 ))
        
        d = q[2] - q[1] + self.l1 - self.l2
        
        pts[0,:]  = np.array([d*0.00 + x2 + self.l2/2,0,0])
        pts[1,:]  = np.array([d*0.20 + x2+self.l2/2,0,0])
        pts[2,:]  = np.array([d*0.25 + x2+self.l2/2,+h,0])
        pts[3,:]  = np.array([d*0.30 + x2+self.l2/2,-h,0])
        pts[4,:]  = np.array([d*0.35 + x2+self.l2/2,+h,0])
        pts[5,:]  = np.array([d*0.40 + x2+self.l2/2,-h,0])
        pts[6,:]  = np.array([d*0.45 + x2+self.l2/2,+h,0])
        pts[7,:]  = np.array([d*0.50 + x2+self.l2/2,-h,0])
        pts[8,:]  = np.array([d*0.55 + x2+self.l2/2,+h,0])
        pts[9,:]  = np.array([d*0.60 + x2+self.l2/2,-h,0])
        pts[10,:] = np.array([d*0.65 + x2+self.l2/2,+h,0])
        pts[11,:] = np.array([d*0.70 + x2+self.l2/2,-h,0])
        pts[12,:] = np.array([d*0.75 + x2+self.l2/2,+h,0])
        pts[13,:] = np.array([d*0.80 + x2+self.l2/2,0,0])
        pts[14,:] = np.array([d*1.00 + x2+self.l2/2,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'k')
                
        return lines_pts , lines_style , lines_color
 
        
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    #sys = SingleMass()
    #sys = TwoMass()
    sys = ThreeMass()
    #sys = FloatingSingleMass()
    #sys = FloatingTwoMass()
    #sys = FloatingThreeMass()
    
    def t2u(t):
        return np.array([t])
    
    sys.t2u   = t2u
    sys.x0[0] = 1.0
    
    sys.compute_trajectory()
    #sys.plot_phase_plane()
    #sys.plot_linearized_bode()
    #sys.plot_linearized_pz_map()
    #sys.plot_trajectory('xu')
    
    sys.lines_plus = True
    #sys.lines_plus = False
    sys.animate_simulation()