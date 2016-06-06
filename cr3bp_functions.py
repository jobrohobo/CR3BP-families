# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:45:20 2016

@author: jbrown
"""

import numpy as np

# equations of motion
def cr3bp_eom(t,stateSTM,mu):
    '''
    stateSTM is a 42 element array; the first 6 elements contain the 
    non-dimensionalized, rotating position and velocity; the remaining elements
    are the state transition matrix. mu is the mass ratio of the two primaries
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
    mu is the mass ratio of the two primaries, state is the non-dimesionalized
    rotating position and velocity
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
    mu is the mass ratio of the two primaries
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
    Lx and Ly are the nondimensionalized position of the libration point, mu is
    the mass ratio of the two primaries, x is the displacement from the 
    libration point of the initial condition (assumes y = z = 0)
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