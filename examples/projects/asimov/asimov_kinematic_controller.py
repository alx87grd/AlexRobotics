###############################################################################
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
from pyro.control import robotcontrollers
from pyro.dynamic import manipulator
from asimov import Asimov
###############################################################################
gain = 1  # Gain du controlleur

asimov = Asimov()  # Asimov
sc_asimov = manipulator.SpeedControlledManipulator.from_manipulator( asimov )

sc_asimov.x0 = np.array([3*np.pi/4, -np.pi/2, np.pi/2])  # Position initiale [q1, q2, q3]

ctl = robotcontrollers.EndEffectorKinematicController(sc_asimov, gain)  # Déclaration du controlleur
ctl.rbar = np.array([0.5, 0.25, 0])  # Cible

closed_loop_robot = ctl + sc_asimov  # Système boucle fermé

closed_loop_robot.plot_trajectory()  # Calcul de la trajectoire
closed_loop_robot.animate_simulation( is_3d=True )