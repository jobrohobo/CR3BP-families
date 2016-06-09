# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:46:00 2016

@author: jbrown
"""
import time #this is only for tic/toc-style functionality while debugging
import numpy as np
from cr3bp_functions import cr3bp_eom, libration_points, Lyap_vel_guess
from scipy.integrate import ode
import matplotlib.pyplot as plt
start = time.time()
import warnings
warnings.filterwarnings("ignore")

#constants and characteristic properties
G = 6.67408e-20 #km^3/kg/s^2
mass1 = 132712200000/G #kg, sun
mass2 = 398600.441800/G #kg, earth
m_star = mass1 + mass2
l_star = 149587457#km, earth sma around sun
t_star = np.sqrt(l_star**3/(G*m_star))
mu = mass2/m_star

#inputs
family_members = 200
x_start = -0.0001 #nondim, first step away from libration point (negative for L1, positive for L2)
x_step = -0.000005 #nondim, all subsequent steps between family members (negative for L1, positive for L2)
x_step_delta = -0.0000001 #nondim
backend_bigprop = 'vode'
backend_littleprop = 'dopri5' #can't use vode with two solvers simultaneously
tol_converge = 1e-10 #nondim
tol_integrate = 1e-12 #nondim
tol_step = 1e-8 #nondim, smallest integration step
final_crossing = 1
t0 = 0
tf = 10
#array to store converged orbit
lyapunovs = []
#find libration point positions/energies
lpstuff = libration_points(mu)
vy_guess = Lyap_vel_guess(lpstuff[0][0],lpstuff[1][0],mu,x_start)
plt.plot(lpstuff[0][0],lpstuff[1][0],'.') #plot L1
STM0 = np.reshape(np.eye(6),36)
#set integrators
bigprop = ode(cr3bp_eom).set_integrator(backend_bigprop,atol=tol_integrate,rtol=tol_integrate)
littleprop = ode(cr3bp_eom).set_integrator(backend_littleprop,atol=tol_integrate,rtol=tol_integrate)
for member in range(0,family_members):
    #initial conditions
    x_step = x_step + x_step_delta
    state0 = [lpstuff[0][0]+x_start+x_step*member,0,0,0,vy_guess,0]
    x0 = np.concatenate((state0, STM0),axis=1)
    for target_crossing in range(1,final_crossing+1):
        error = 1
        while np.abs(error) > tol_converge:
            #integrate
            bigprop.set_initial_value(x0,t0).set_f_params(mu)
            t = [t0]
            x = [x0[0]]
            y = [x0[1]]
            z = [x0[2]]
            current_crossing = 0
            last_y = np.sign(x0[4])
            while bigprop.successful() and current_crossing < target_crossing:
                #step
                current_state = bigprop.integrate(tf,step=True)
                #check for crossing
                if np.sign(current_state[1]) != np.sign(last_y):
                    current_crossing += 1
                    #littleprop to find actual crossing
                    dt = (t[-1]-t[-2])
                    while np.abs(current_state[1]) > tol_converge and abs(dt) > tol_step:
                        dt = -dt/2
                        last_y = current_state[1]
                        littleprop.set_initial_value(current_state,bigprop.t).set_f_params(mu)
                        current_state = littleprop.integrate(littleprop.t+dt)
                        #take another step if we didn't cross on the first one
                        if np.sign(current_state[1]) == np.sign(last_y):
                            current_state = littleprop.integrate(littleprop.t+dt)
                    #save correct time
                    t.append(littleprop.t)
                else:
                    #save correct time
                    t.append(bigprop.t)
                #save data to arrays
                x.append(current_state[0])
                y.append(current_state[1])
                z.append(current_state[2])
                last_y = current_state[1]
#            #plot
#            plt.plot(x,y,'.-')
            #update initial conditions
            error = current_state[3]
            #simple
#            dvy = error/current_state[28]
#            print error,dvy
#            x0[4] = x0[4] - dvy
            #intermediate
            r1 = np.sqrt( (current_state[0]+mu)**2 + current_state[1]**2 + current_state[2]**2 )
            r2 = np.sqrt( (current_state[0]-1+mu)**2 + current_state[1]**2 + current_state[2]**2 )
            ax = 2*current_state[4] + current_state[0] - (1-mu)*(current_state[0]+mu)/r1**3 - mu*(current_state[0]-1+mu)/r2**3
            dvy = (current_state[4]*error)/(current_state[28]*current_state[4]-current_state[16]*ax)
            #dvy = current_state[4]/ax*current_state[3]/(current_state[4]/ax*current_state[28]-current_state[16])
#            print member,target_crossing,error,dvy,time.time()-start
            x0[4] = x0[4] - dvy
            #matrix
#            X = np.array([[current_state[4]],[r.t]])
#            FX = np.array([[current_state[1]],[current_state[3]]])
#            DF = np.array([[current_state[16],current_state[4]],[current_state[28],ax]])
#            update = np.zeros((2,1))
#            update = np.mat(X) - np.mat(np.transpose(DF))*np.linalg.inv(np.mat(DF)*np.mat(np.transpose(DF)))*np.mat(FX)
#            print error, update[0]-X[0]
#            x0[4] = update[0]
            
    #save converged orbit
    lyapunovs.append([t,x,y,z])
    #output
    print member,time.time()-start
    plt.plot(lyapunovs[-1][1],lyapunovs[-1][2],'g-')
    plt.plot(lyapunovs[-1][1],[-1*i for i in lyapunovs[-1][2]],'g-')
    #update guess for next member in family
    vy_guess = x0[4]
print time.time()-start