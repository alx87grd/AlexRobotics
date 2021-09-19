###############################################################################
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
from asimov import Asimov
from pyro.control import nonlinear
###############################################################################

asimov = Asimov()  # Asimov
x0 = np.array([-np.pi/4, -3*np.pi/4, np.pi/2, 0, 0, 0])  # Position initiale

ctl = nonlinear.ComputedTorqueController(asimov)  # Déclaration du controlleur
ctl.rbar = np.array([0.5, 0.25, 0])  # Cible
ctl.w0 = 2
ctl.zeta = 1

closed_loop_robot = ctl + asimov  # Système boucle fermé

closed_loop_robot.plot_trajectory(x0, 5)  # Calcul de la trajectoire
closed_loop_robot.sim.plot('x')  # Affichage des états
closed_loop_robot.sim.plot('u')  # Affichage des commandes
closed_loop_robot.animate_simulation(1, True)  # Animation et enregistrement

plt.show()
