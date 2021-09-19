# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:01:07 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
import advanced_vehicles
import test_vehicle_controllers
###############################################################################

# "Fake" controller - Varying inputs (delta, T_f, T_r) throughout time (change in linear.py)
ctl = test_vehicle_controllers.fullDynVoltInputs()
# Vehicule dynamical system
sys = advanced_vehicles.FullDynamicBicycleModelwithVoltInput()
# Add the inputs to the dynamical system
cl_sys = ctl + sys
# Plot open-loop behavior (ex: np.array[intial_conditions], time_of_simulation)
cl_sys.x0 =  np.array([1,0,0,0,0,0,1/sys.r])
cl_sys.compute_trajectory( 50 )
cl_sys.plot_trajectory('xu')

# Rebuild x,u and t from simulation
x = cl_sys.traj.x
u = cl_sys.traj.u 
t = cl_sys.traj.t    

## Compute tire forces and slip angles for graphical purpose  
#F_nf = sys.mass*sys.g*sys.b/(sys.b+sys.a)
#F_nr = sys.mass*sys.g*sys.a/(sys.b+sys.a)
#mu = sys.mu
#max_F_f = F_nf*mu
#max_F_r = F_nr*mu
#max_alpha_stat = sys.max_alpha_lin
#slip_ratio_f = max_F_f/(max_alpha_stat) #0.12 is the max slipping angle in rad before kinetic slip
#slip_ratio_r = max_F_r/(max_alpha_stat) 
## Simulation of road-tire behavior according to slip-ratios
#max_slip_stat = sys.max_slip_lin
#slip_slope_f = max_F_f/max_slip_stat
#slip_slope_r = max_F_r/max_slip_stat
## Init. vectors for display purposes
#F_yf = np.zeros(len(t))
#F_yr = np.zeros(len(t))
#slip_f = np.zeros(len(t))
#slip_r = np.zeros(len(t))
#maxF_f = np.zeros(len(t))
#maxF_r = np.zeros(len(t))
#F_xr = np.zeros(len(t))
#long_slip_r = np.zeros(len(t))
## Iterate through time to compute slip angles, slip ratios, lateral forces and longitudinal forces
#for i in range(len(t)-1):
#    
#    if(x[i,0] == 0 and x[i,6] == 0):
#        F_xr[i] = 0
#        long_slip_r[i] = 0
#    else:
#        long_slip_r[i] = (x[i,6]*sys.r-x[i,0])/max(x[i,0],x[i,6]*sys.r)
#        if (long_slip_r[i]<-0.03):
#            F_xr[i] = -max_F_r
#        elif(long_slip_r[i]>0.03):
#            F_xr[i] = max_F_r
#        else:
#            F_xr[i] = slip_slope_r*long_slip_r[i]
#    maxF_f[i] = max_F_f
#    maxF_r[i] = max_F_r
#
## Plot forces
#figsize   = (7, 4)
#dpi       = 100
#plt.figure(2, figsize=figsize, dpi=dpi)
#plt.title('Longitudinal forces for the longitudinal dynamic\n model with steering angle and torque as inputs', fontsize=20)
#plt.plot(t[:-1], F_xr[:-1], label = 'F_xr')
#plt.plot(t[:-1], maxF_f[:-1],'--', label = 'Max front force')
#plt.plot(t[:-1], maxF_r[:-1],'--', label = 'Max rear force')
#plt.plot(t[:-1], -maxF_r[:-1],'--', label = 'Max rear force')
#plt.legend(fontsize = '15')
#plt.xlabel('Temps (s)')
#plt.ylabel('Force (N)')
#plt.show()
#
#
## Plot trajectory of the vehicle's CG
#slip = 1
#t_start_slip = 'No slip' #No slip
#plt.figure(3,figsize=figsize, dpi=dpi)
#for i in range(len(t)-1):
#    if (F_xr[i]==max_F_r):
#        if (slip == 1):            
#            t_start_slip = i
#            slip = 0
#        else:
#            t_stop_slip = i
#    else:
#        pass
#if (t_start_slip == 'No slip'):
#    plt.plot(x[:,4],x[:,5],'-b', label = 'No slip')  
#else:
#    plt.plot((x[t_start_slip:t_stop_slip,4]),(x[t_start_slip:t_stop_slip,5]),'-r',label='Slip')
#    plt.plot((x[0:t_start_slip,4]),(x[0:t_start_slip,5]),'-b', label='No slip')
#    plt.plot((x[t_stop_slip:len(t)-1,4]),(x[t_stop_slip:len(t)-1,5]),'-b')    
#plt.legend(fontsize ='15')
#plt.show()
#
## Plot slip_ratios according to time
#plt.figure(4, figsize=figsize, dpi=dpi)
#plt.title('Slip ratios', fontsize=20)
#plt.plot(t[:-1], long_slip_r[:-1], label = 'Sx_r')
#plt.legend(fontsize = '15')
#plt.xlabel('Temps (s)')
#plt.ylabel('Slip ratio')
#plt.show()

# Animate the simulation
cl_sys.plot_trajectory('x')
cl_sys.plot_trajectory('u')
cl_sys.animate_simulation()

