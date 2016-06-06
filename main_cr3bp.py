# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:46:00 2016

@author: jbrown
"""
import numpy as np
from cr3bp_functions import cr3bp_eom, libration_points, Lyap_vel_guess
from scipy.integrate import ode
import matplotlib.pyplot as plt

#constants and characteristic properties
G = 6.67408e-20 #km^3/kg/s^2
mass1 = 132712200000/G #kg, sun
mass2 = 398600.441800/G #kg, earth
m_star = mass1 + mass2
l_star = 149587457#km, earth sma around sun
t_star = np.sqrt(l_star**3/(G*m_star))
mu = mass2/m_star

#inputs
x_start = -0.0001 #nondim, first step away from libration point (negative for L1, positive for L2)
x_step = -0.000005 #nondim, all subsequent steps between family members (negative for L1, positive for L2)
tol_converge = 1e-11 #nondim
tol_integrate = 1e-12 #nondim
final_crossing = 1
t0 = 0
tf = 5.349501906
dt = 0.01

#find libration point positions/energies
lpstuff = libration_points(mu)
vy_guess = Lyap_vel_guess(lpstuff[0][0],lpstuff[1][0],mu,x_start)
plt.plot(lpstuff[0][0],lpstuff[1][0],'.') #plot L1
STM0 = np.reshape(np.eye(6),36)
for member in range(0,5):
    #initial conditions
    state0 = [lpstuff[0][0]+x_start+x_step*member,0,0,0,vy_guess,0]
    x0 = np.concatenate((state0, STM0),axis=1)
    for target_crossing in range(1,final_crossing+1):
        error = 1
        while np.abs(error) > tol_converge:
            #integrate
            r = ode(cr3bp_eom).set_integrator('dopri5',atol=tol_integrate,rtol=tol_integrate)
            r.set_initial_value(x0,t0).set_f_params(mu)
            t,x,y = [],[],[]
            t.append(t0)
            x.append(x0[0])
            y.append(x0[1])
            current_crossing = 0
            last_y = np.sign(x0[4])
            while r.successful() and current_crossing < target_crossing:
                #step
                state_array = r.integrate(r.t+dt)
                #check for crossing
                if np.sign(state_array[1]) != np.sign(last_y):
                    current_crossing += 1
                    #bisect to find actual crossing
                    while np.abs(state_array[1]) > tol_converge:
                        dt = -dt/2
                        last_y = state_array[1]
                        state_array = r.integrate(r.t+dt)
                        #take another step if we didn't cross on the first one
                        while np.sign(state_array[1]) == np.sign(last_y):
                            state_array = r.integrate(r.t+dt)
                    #reset step
                    dt = 0.01
                #save data to arrays
                t.append(r.t)
                x.append(state_array[0])
                y.append(state_array[1])
                last_y = state_array[1]
            #plot
            plt.plot(x,y)
            #update initial conditions
            error = state_array[3]
            #simple
#            dvy = error/state_array[28]
#            print error,dvy
#            x0[4] = x0[4] - dvy
            #intermediate
            r1 = np.sqrt( (state_array[0]+mu)**2 + state_array[1]**2 + state_array[2]**2 )
            r2 = np.sqrt( (state_array[0]-1+mu)**2 + state_array[1]**2 + state_array[2]**2 )
            ax = 2*state_array[4] + state_array[0] - (1-mu)*(state_array[0]+mu)/r1**3 - mu*(state_array[0]-1+mu)/r2**3
            dvy = (state_array[4]*error)/(state_array[28]*state_array[4]-state_array[16]*ax)
            #dvy = state_array[4]/ax*state_array[3]/(state_array[4]/ax*state_array[28]-state_array[16])
            print member,target_crossing,error,dvy
            x0[4] = x0[4] - dvy
            #matrix
#            X = np.array([[state_array[4]],[r.t]])
#            FX = np.array([[state_array[1]],[state_array[3]]])
#            DF = np.array([[state_array[16],state_array[4]],[state_array[28],ax]])
#            update = np.zeros((2,1))
#            update = np.mat(X) - np.mat(np.transpose(DF))*np.linalg.inv(np.mat(DF)*np.mat(np.transpose(DF)))*np.mat(FX)
#            print error, update[0]-X[0]
#            x0[4] = update[0]
            
    #update guess for next member in family
    vy_guess = x0[4]