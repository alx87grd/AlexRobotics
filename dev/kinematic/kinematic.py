# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 17:07:22 2012

@author: gira2403
"""


"""
###########################################################
# Modules
###########################################################
"""

# Numpy
import numpy as np
from numpy.linalg import norm


"""
###########################################################
##  Constantes
###########################################################
"""

epsilon = 1e-12


"""
###########################################################
##  Class
###########################################################
"""
        
###########################################################
##  3D Vector
###########################################################

class Vector:
    """ 3D Vector """
    
    def __init__(self, x = 0, y = 0, z = 0):
        """ Default is a null vector """
        
        self.x = x
        self.y = y
        self.z = z
        
        """ Numpy matrix (COLUNM) """
        self.col  = np.matrix([[x],[y],[z]])
        
        """ Numpy skew matrix """
        self.skew = np.matrix([[0,-z,y],[z,0,-x],[-y,x,0]])
        
        """ Vectorial norm """
        self.norm = norm(self.col)
        
    
    ##################################################
    def __call__(self):
        
        print('3D Vector : [',self.x,';',self.y,';',self.z,']')
        
        
    #################################
    def __add__(self,other):
        """ Vectorial addition """
        
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        
        return Vector(x,y,z)
        
        
    #################################    
    def __sub__(self,other):
        """ Vectorial substraction """
        
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        
        return Vector(x,y,z)
        
        
    #################################    
    def __neg__(self):
        """ Vectorial inversion """
        
        x = - self.x
        y = - self.y
        z = - self.z
        
        return Vector(x,y,z)
        
        
    #################################
    def __mul__(self,other):
        """ Scalar multiplication with a scalar or dot product with a vector """
        
        """ Check if its a scalar multiplication or a dot product """
        isvector = isinstance(other,Vector)
        
        ##############
        if isvector:
            
            # Dot product
            ans = other.col.T * self.col
            
            return ans[0,0]
        
        ##############
        else:
            
            # Scalar multiplication
            x = self.x * other
            y = self.y * other
            z = self.z * other
        
            return Vector(x,y,z)
            
            
    #################################
    def __pow__(self,other):
        """ Vectorial product """
        
        col = self.skew * other.col
        
        return col2vec(col)
        
        
    #################################
    def normalize(self):
        """ Return normalized vector """
        
        ############################
        if self.norm > 0 :
            
            direction = self * ( 1 / self.norm )
        
        else :
            print('Kinematic warning : not able to normalize a zero vector')
            direction = Vector(0,0,0)
        
        
        return direction
        
        
    #################################
    def copy(self):
        """ Return a copy of the current vector """
        
        copy = self * 1
        
        return copy
        


###########################################################
##   Rotation Matrix
###########################################################

class RotationMatrix:
    """ Matrix representation of a 3D rotation """
    
    ###########################
    def __init__(self,  matrix = np.matrix(np.eye(3,3)) ):
        """ Default is a identity matrix """
        
        self.C = matrix
        
        
    #######################
    def __call__(self):
        
        print('Rotation Matrix : \n', self.C)
        
        
    #################################
    def __mul__(self,other):
        """ Matrix multiplication """
        
        """ Check if other is RotationMatrix or a vector """
        isvector = isinstance(other,Vector)
        ismatrix = isinstance(other,RotationMatrix)
        
        ##################
        if isvector:
            
            col = self.C * other.col
            
            return col2vec(col)
        
        ##################
        elif ismatrix:
            
            mat = self.C * other.C
            
            return RotationMatrix(mat)
        
        ##################
        else:
            """ Scale the rotation arround the same axis with a scalar """
            
            new_rotation =  self.toAngleAxis() * other
            
            return AngleAxis2RotationMatrix(new_rotation)
        
        
    #################################    
    def __neg__(self):
        """ Inverse Matrix """
        
        return RotationMatrix(self.C.T)
        
        
    #################################    
    def toAngleAxis(self):
        """ Compute equivalent Angle-Axis representation """
        
        return RotationMatrix2AngleAxis(self)    
    
    
    #################################    
    def toQuaternion(self):
        """ Compute equivalent quaternion representation """
        
        return RotationMatrix2Quaternion(self)
        
        
    #################################    
    def toRotationVector(self):
        """ Compute equivalent Rotation Vector """
        
        a = self.toAngleAxis()
        
        return a.toRotationVector()
        

###########################################################
##  Angle-Axis 3D rotation
###########################################################

class AngleAxis:
    """ Angle-Axis representation of a rotation in 3D """
    
    ################
    def __init__(self, rad = 0, axis = Vector(1,0,0) ):
        """ Default is a rotation of 0 degree arround x """
        
        self.rad = rad
        self.deg = np.rad2deg(rad)
        
        self.axis = axis
        
        
    ################
    def __call__(self):
        """ Print Angle Axis """
        
        print('AngleAxis: \n deg : ',self.deg,' axis : [',self.axis.x,';',self.axis.y,';',self.axis.z,']')
        
    
    #################################
    def __mul__(self,other):
        """ Scale the rotation around the same axis """
        
        rad  = self.rad * other
        axis = self.axis.copy()
        
        return AngleAxis(rad, axis)
        
        
    ################
    def toRotationMatrix(self):
        """ Convert to 3x3 rotation matrix """
        
        return AngleAxis2RotationMatrix(self)
        
        
    ################
    def toQuaternion(self):
        """ Convert to Quaternion """
        
        return AngleAxis2Quaternion(self)
        
        
    ################
    def toRotationVector(self):
        """ Convert to Rotation Vector """
        
        vector = self.axis * self.rad
        
        return RotationVector( vector.x , vector.y , vector.z )
        
        
###########################################################
##  Vecteur Rotation Rodrigues ( axis * angle )
###########################################################

class RotationVector(Vector):
    """ 3D Vector representation of a rotation in 3D (angle axis : axis * angle) """
    
    ####################
    def __call__(self):
        """ Print Rotation Vector """
        
        print('3D Rotation Vector : [',self.x,';',self.y,';',self.z,']')
        
        
    ####################
    def toAngleAxis(self):
        """ Convert to Angle-Axis """
        
        rad  = self.norm
        axis = self.normalize()
        
        return AngleAxis(rad,axis)
        
    ################
    def toRotationMatrix(self):
        """ Convert to 3x3 rotation matrix """
        
        a = self.toAngleAxis()
        
        return AngleAxis2RotationMatrix(a)
        
        
    ################
    def toQuaternion(self):
        """ Convert to Quaternion """
        
        a = self.toAngleAxis()
        
        return AngleAxis2Quaternion(a)
        

###########################################################
##  Quaternion
###########################################################

class Quaternion:
    """ Quaternion representation of a 3D rotation """
    
    #################
    def __init__(self, e = Vector(), n = 1 ):
        """ Default is a zero rotation """
        
        self.e = e
        self.n = n
        
        
    ################
    def __call__(self):
        """ Print Quaternion """
        
        print('Quaternion: \n e : [',self.e.x,';',self.e.y,';',self.e.z,']  n : ',self.n)
        
        
    #################################
    def __mul__(self,other):
        """ Quaternion multiplication or vector rotation with the Quaternion """
        
        
        """ Check if other is RotationMatrix or a vector """
        isvector     = isinstance(other,Vector)
        isquaternion = isinstance(other,Quaternion)
        
        ################
        if isquaternion:
            """ Quaternion Multiplication """
            # Q = Qa * Qb
            
            na = self.n
            nb = other.n
            
            ea = self.e
            eb = other.e
            
            n = na * nb - ea * eb                     #  * operator = scalar product for vector
            e = (eb * na) + (ea * nb) + (ea ** eb)    #  ** operator = vectorial product for vector
            
            # Attention définition right handed, donc R2*R1 = Q1*Q2
            
            """
            # 4x4 matrix to compute quaternion multiplication
            qskew = np.matrix(np.zeros((4,4)))
            
            # Identitie matrix
            I = np.matrix(np.eye(3,3))
            
            qskew[0:3,0:3] = self.n * I - self.e.skew
            qskew[3,0:3]   =  - self.e.col.T
            qskew[0:3,3]   =  self.e.col
            qskew[3,3]     = self.n
            """
            
            return Quaternion(e,n)
        
        ################
        if isvector:
            """ Rotate vector computing the rotation matrix """
            
            new_vector = self.toRotationMatrix() * other
            
            return new_vector
            
        ############
        else:
            """ Scale the rotation arround the same axis (other * angle arround the same axis) """
            
            new_rotation =  self.toAngleAxis() * other
            
            return AngleAxis2Quaternion(new_rotation)
            
    
    #################################
    def __neg__(self):
        """ Inverse Quaternion """
        
        return Quaternion(-self.e,self.n)
    
    
    #################################    
    def toRotationMatrix(self):
        """ Compute equivalent rotation matrix """
        
        return Quaternion2RotationMatrix(self)
        
        
    #################################    
    def toAngleAxis(self):
        """ Compute equivalent Angle-Axis """
        
        return Quaternion2AngleAxis(self)
        
        
    #################################    
    def toRotationVector(self):
        """ Compute equivalent Rotation Vector """
        
        a = self.toAngleAxis()
        
        return a.toRotationVector()
        


"""
##############################################################################
#############         Functions    ###########################################         
##############################################################################
"""

########################################
def col2vec(col):
    """ 
    Create a vector class from a numpy matrix 
    
    col = 
    matrix([[1],
            [2],
            [1]])
    
    """
    
    x = col[0,0]
    y = col[1,0]
    z = col[2,0]
    
    return Vector(x,y,z)
    
    
########################################
def list2vec(l):
    """ 
    Create a vector class from a list
    
    l = [x,y,z]
    
    """
    
    x = l[0]
    y = l[1]
    z = l[2]
    
    return Vector(x,y,z)
    
    
########################################
def euler2RotationMatrix( teta_1 , teta_2 , teta_3 ):
    """ Convert 3 euler angle to a 321 rotation matrix """

    """ Convert degree to radian """
    r1 = np.deg2rad(teta_1)
    r2 = np.deg2rad(teta_2)
    r3 = np.deg2rad(teta_3)
    
    """ Compute cosinus """
    c1 = np.cos(r1)
    c2 = np.cos(r2)
    c3 = np.cos(r3)
    
    """ Compute sinus """
    s1 = np.sin(r1)
    s2 = np.sin(r2)
    s3 = np.sin(r3)
    
    """ Compute rotation matrix """
    R1 = np.matrix([[1,0,0],[0,c1,s1],[0,-s1,c1]])
    R2 = np.matrix([[c2,0,-s2],[0,1,0],[s2,0,c2]])
    R3 = np.matrix([[c3,s3,0],[-s3,c3,0],[0,0,1]])
    
    """ Rotation 3-2-1 """
    C = R1 * R2 * R3
    
    return RotationMatrix(C)
    
    
###########################################
def RotationMatrix2AngleAxis(RM):
    """ Convert a rotation matrix to a angle axis class """
    
    C = RM.C
    
    """ Compute Angle """
    cos = (C.trace()[0,0] - 1) * 0.5
    
    rad = np.arccos(cos)
    
    sin = np.sin(rad)
    
    """ Compute axis of rotation """
    ##############
    if ( abs(rad - np.pi)  > epsilon ) and ( abs(rad + np.pi)  > epsilon ):
    
        ax = ( C[1,2] - C[2,1] ) / (2 * sin)
        ay = ( C[2,0] - C[0,2] ) / (2 * sin)
        az = ( C[0,1] - C[1,0] ) / (2 * sin)
        
    ##############
    else:
        
        print('\n RotationMatrix2AngleAxis : bad matrix : to do!!!')
    
    axis = Vector(ax,ay,az)
    
    return AngleAxis(rad,axis)


###########################################
def AngleAxis2Quaternion(AA):
    """ Convert a angle axis class to a Quaternion """
    
    """ Vector """
    e = AA.axis * np.sin( AA.rad * 0.5 )
    
    """ Scalar """
    n = np.cos( AA.rad * 0.5 )
    
    return Quaternion(e,n)
    
    
###########################################
def AngleAxis2RotationMatrix(AA):
    """ Convert a angle axis class to a Rotation Matrix """
    
    a   = AA.axis
    rad = AA.rad

    """ Eye matrix """
    I = np.matrix(np.eye(3,3))
    
    cos = np.cos(rad)
    sin = np.sin(rad)
    
    """ Compute Matrix """
    C = cos * I + (1 - cos) * a.col * a.col.T - sin * a.skew
    
    return RotationMatrix(C)


###########################################
def Quaternion2RotationMatrix(Q):
    """ Convert a Quaternion to a Rotation Matrix  """
    
    n = Q.n
    e = Q.e
    
    """ Eye matrix """
    I = np.matrix(np.eye(3,3))
    
    C = I * ( n**2 - e.norm**2 ) + 2 * e.col * e.col.T - 2 * n * e.skew
    
    return RotationMatrix(C)
    

###########################################
def Quaternion2AngleAxis(Q):
    """ Convert a Quaternion to a angle axis """
    
    n = Q.n
    e = Q.e
    
    """ Axis """
    axis = e * ( 1 / e.norm )
    
    """ Angle """
    rad = 2 * np.arccos(n)
    
    return AngleAxis(rad,axis)

    
###########################################
def RotationMatrix2Quaternion(RM):
    """ Convert a Quaternion to a Rotation Matrix  """
    
    C = RM.C
    
    trace = C.trace()[0,0]
    
    if trace < epsilon :
        print(' Warning : trace of the matrix is near zero ')
    
    n = 0.5 * np.sqrt( trace + 1 )
    
    ex = C[1,2] - C[2,1]
    ey = C[2,0] - C[0,2]
    ez = C[0,1] - C[1,0]
    
    e = Vector(ex,ey,ez) * ( 1 / ( 4 * n ) )
    
    return Quaternion(e,n)
    
    
    

"""
##############################################################################
#############         MAIN         ###########################################         
##############################################################################
"""


if __name__ == '__main__':
         
    V = Vector(10,10,10)
    
    C1 = euler2RotationMatrix(0,0,90)
    C2 = euler2RotationMatrix(90,0,0)
    
    Q1 = C1.toQuaternion()
    Q2 = C2.toQuaternion()
    
    C21 = C1 * C2
    
    Q21 = Q2 * Q1
  
    