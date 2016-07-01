# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:45:20 2016

@author: jbrown
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

# equations of motion
def cr3bp_eom(t,stateSTM,mu):
    '''
    stateSTM = 42 element array; the first 6 elements contain the 
        non-dimensionalized, rotating position and velocity; the remaining 
        elements are the state transition matrix. 
    mu = mass ratio of the two primaries
    '''
    #intermediate variables
    x = stateSTM[0]
    y = stateSTM[1]
    z = stateSTM[2]
    vx = stateSTM[3]
    vy = stateSTM[4]
    vz = stateSTM[5]
    
    #range between s/c and each primary
    r1 = np.sqrt( (x+mu)**2 + y**2 + z**2 )
    r2 = np.sqrt( (x-1+mu)**2 + y**2 + z**2 )
    
    #integrate spacecraft
    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
    az = -(1-mu)*z/r1**3 - mu*z/r2**3
    
    #build A matrix
    Uxx = 1 - (1-mu)/r1**3 - mu/r2**3 + (3*(1-mu)*(x+mu)**2)/r1**5 + (3*mu*(x-1+mu)**2)/r2**5;
    Uyy = 1 - (1-mu)/r1**3 - mu/r2**3 + (3*(1-mu)* y    **2)/r1**5 + (3*mu* y      **2)/r2**5;
    Uzz =   - (1-mu)/r1**3 - mu/r2**3 + (3*(1-mu)* z    **2)/r1**5 + (3*mu* z      **2)/r2**5;
    Uxy = (3*(1-mu)*(x+mu)*y)/r1**5 + (3*mu*(x-1+mu)*y)/r2**5;
    Uyx = Uxy;
    Uxz = (3*(1-mu)*(x+mu)*z)/r1**5 + (3*mu*(x-1+mu)*z)/r2**5;
    Uzx = Uxz;
    Uyz = (3*(1-mu)*y*z)/r1**5 + 3*mu*y*z/r2**5;
    Uzy = Uyz;
    A = np.zeros((6,6))
    A[0:3,3:6] = np.eye(3)
    A[3:6,0:3] = np.array([[Uxx,Uxy,Uxz],[Uyx,Uyy,Uyz],[Uzx,Uzy,Uzz]])
    A[3:6,3:6] = np.array([[0,2,0],[-2,0,0],[0,0,0]])
    
    #integrate STM
    STM = np.reshape(stateSTM[6:],(6,6))
    STMdot = np.zeros((6,6))
    STMdot = np.dot(A,STM)
    
    #output
    stateSTMdot = np.concatenate((np.array([vx,vy,vz,ax,ay,az]),STMdot.reshape(36)),axis=1)
    
    return stateSTMdot
    
# Jacobi constant
def Jacobi(mu,state):
    '''
    mu = mass ratio of the two primaries
    state = non-dimesionalized rotating position and velocity
    '''    
    #array variables
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    
    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)
    vmag = vx**2 + vy**2 + vz**2
    C = x**2 + y**2 + 2*(1-mu)/r1 + 2*mu/r2 - vmag**2
    return C
    
# libration point positions and energies
def libration_points(mu):
    '''
    mu = mass ratio of the two primaries
    '''
    tol = 1e-12
    x,y,C = [],[],[]
    
    #L1
    xold = 0.99
    error = 1
    while error > tol:
        f = xold - (1-mu)/(mu+xold)**2 + mu/(xold-1+mu)**2
        fprime = 1 + 2*(1-mu)*(mu+xold)/(mu+xold)**4 - 2*mu*(xold-1+mu)/(xold-1+mu)**4
        xnew = xold - f/fprime
        error = abs(xnew-xold)
        xold = xnew
    x.append(xold)
    y.append(0.)
    C.append(Jacobi(mu,[x[0],y[0],0,0,0,0]))
    
    #L2
    xold = 1.01
    error = 1
    while error > tol:
        f = xold - (1-mu)/(mu+xold)**2 - mu/(xold-1+mu)**2
        fprime = 1 + 2*(1-mu)*(mu+xold)/(mu+xold)**4 + 2*mu*(xold-1+mu)/(xold-1+mu)**4
        xnew = xold - f/fprime
        error = abs(xnew-xold)
        xold = xnew
    x.append(xold)
    y.append(0.)
    C.append(Jacobi(mu,[x[1],y[1],0,0,0,0]))
    
    #L3
    xold = -1
    error = 1
    while error > tol:
        f = xold + (1-mu)/(mu+xold)**2 + mu/(xold-1+mu)**2
        fprime = 1 - 2*(1-mu)*(mu+xold)/(mu+xold)**4 - 2*mu*(xold-1+mu)/(xold-1+mu)**4
        xnew = xold - f/fprime
        error = abs(xnew-xold)
        xold = xnew
    x.append(xold)
    y.append(0.)
    C.append(Jacobi(mu,[x[2],y[2],0,0,0,0]))
    
    #L4
    x.append(0.5-mu)
    y.append(np.sqrt(3)/2)
    C.append(Jacobi(mu,[x[3],y[3],0,0,0,0]))
    
    #L5
    x.append(0.5-mu)
    y.append(np.sqrt(3)/2)
    C.append(Jacobi(mu,[x[4],y[4],0,0,0,0]))
    
    return [x,y,C]
    
#Lyapunov guess for initial velocity
def Lyap_vel_guess(Lx,Ly,mu,x):
    '''
    Lx, Ly = nondimensionalized position of the libration point
    mu = mass ratio of the two primaries
    x = displacement from the libration point of the initial condition (assumes y = z = 0)
    '''
    r1 = np.sqrt((Lx+mu)**2 + Ly**2)
    r2 = np.sqrt((Lx-1+mu)**2 + Ly**2)
    Uxx = 1 - (1-mu)/r1**3 - mu/r2**3 + (3*(1-mu)*(Lx+mu)**2)/r1**5 + (3*mu*(Lx-1+mu)**2)/r2**5;
    Uyy = 1 - (1-mu)/r1**3 - mu/r2**3 + (3*(1-mu)* Ly    **2)/r1**5 + (3*mu* Ly      **2)/r2**5;
    
    #coefficients for linearized Lyapunov orbit
    B1 = 2 - (Uxx + Uyy)/2
    B2 = np.sqrt(-Uxx*Uyy)
    s = np.sqrt(B1 + np.sqrt(B1**2 + B2**2))
    B3 = (s**2 + Uxx)/(2*s)
    
    vy = -B3*x*s
    return vy    
    
#target orbit which achieves perpendicular crossing
def target_orbit(target_crossing,x0,t0,tf,mu,backend_bigprop,backend_littleprop,tol_integrate,tol_step,tol_converge):
    '''
    target_crossing = nth crossing to target a perpendicular crossing
    x0 = 42 element initial state; the first 6 elements contain the 
        non-dimensionalized, rotating position and velocity; the remaining 
        elements are the state transition matrix.
    t0 = initial time
    tf = final time (this is typically going to be larger than the actual 
        desired propagation time)
    mu = mass ratio of the two primaries
    backend_bigprop = integrator to use when taking large steps
    backend_littleprop = integrator to use when bisecting the crossing (if 
        bigprop uses vode, this must use a different integrator)
    tol_integrate = integration tolerance
    tol_step = smallest step size that the integrator will use
    tol_converge = tolerance to determine whether crossing is perpendicular
    '''
    bigprop = ode(cr3bp_eom).set_integrator(backend_bigprop,atol=tol_integrate,rtol=tol_integrate)
    littleprop = ode(cr3bp_eom).set_integrator(backend_littleprop,atol=tol_integrate,rtol=tol_integrate)
    error = 1
    iteration = 0
    while np.abs(error) > tol_converge:
        #integrate
        bigprop.set_initial_value(x0,t0).set_f_params(mu)
        t = [t0]
        x = [x0]
        current_crossing = 0
        last_y = np.sign(x0[4])
        while bigprop.successful() and current_crossing < target_crossing:
            #step
            current_state = bigprop.integrate(tf,step=True)
            #check for crossing
            if np.sign(current_state[1]) != np.sign(last_y):
                current_crossing += 1
                #littleprop to find actual crossing
                if current_crossing == target_crossing:
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
            x.append(current_state)
            last_y = current_state[1]
        #targeting algorithm to update initial conditions
        error = current_state[3]
        r1 = np.sqrt( (current_state[0]+mu)**2 + current_state[1]**2 + current_state[2]**2 )
        r2 = np.sqrt( (current_state[0]-1+mu)**2 + current_state[1]**2 + current_state[2]**2 )
        ax = 2*current_state[4] + current_state[0] - (1-mu)*(current_state[0]+mu)/r1**3 - mu*(current_state[0]-1+mu)/r2**3
        X = np.array([[x0[4]],[t[len(t)-1]]])
        FX = np.array([[current_state[1]],[current_state[3]]])
        DF = np.array([[current_state[16],current_state[4]],[current_state[28],ax]])
        update = np.zeros((2,1))
        update = np.mat(X) - np.mat(np.transpose(DF))*np.linalg.inv(np.mat(DF)*np.mat(np.transpose(DF)))*np.mat(FX)
        x0[4] = update[0]
        #increase perpendicular crossing tolerance if stuck
        if iteration > 10:
            tol_converge = 10*tol_converge
            iteration = -1
        iteration += 1
        
    return t,x

#check imaginary components of eigenvalues, and set to 0 if below tolerance
def check_eigenvalues(eigval,tol_eig):
    '''
    eigval = list of eigenvalues
    tol_eig = tolerance for imaginary component of eigenvalues; anything less 
        will be set to 0
    '''
    for i in range(len(eigval)):
        if np.abs(np.imag(eigval[i])) < tol_eig:
            eigval[i] = np.real(eigval[i])
    return eigval