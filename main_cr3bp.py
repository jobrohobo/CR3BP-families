# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:46:00 2016

@author: jbrown
"""
import time #this is only for tic/toc-style functionality while debugging
import numpy as np
from cr3bp_functions import libration_points, Lyap_vel_guess, target_orbit, check_eigenvalues
import matplotlib.pyplot as plt
import sys
Start = time.time()

#constants and characteristic properties
G = 6.67408e-20 #km^3/kg/s^2
Mass1 = 132712200000/G #kg, sun
Mass2 = 398600.441800/G #kg, earth
MStar = Mass1 + Mass2
LStar = 149587457#km, earth sma around sun
TStar = np.sqrt(LStar**3/(G*MStar))
mu = Mass2/MStar

#inputs
FamilyMembers = 1000
PercentToPlot = 2 #saves and plots x% of the orbits generated; others are not saved to limit memory usage 
XStart = -0.0001 #nondim, first step away from libration point (negative for L1, positive for L2)
XStep = -0.000005 #nondim, all subsequent steps between family members (negative for L1, positive for L2)
BackendBigprop = 'vode'
BackendLittleprop = 'dopri5' #can't use vode with two solvers simultaneously
TolConverge = 1e-10 #nondim
TolIntegrate = 1e-12 #nondim
TolStep = 1e-8 #nondim, smallest integration step
TolEig = 1e-3 #if imaginary component of eigenvalue is less than this, it will be set to 0
FinalCrossing = 2 #1 for half rev, 2 for full rev
t0 = 0
tf = 10
#find libration point positions/energies
LibrationPointParams = libration_points(mu)
VyGuess = Lyap_vel_guess(LibrationPointParams[0][0],LibrationPointParams[1][0],mu,XStart)
STM0 = np.reshape(np.eye(6),36)
#array to store converged orbit
Lyapunovs = []
#create lypunov orbits
for Member in range(0,FamilyMembers):
    #initial conditions
    State0 = [LibrationPointParams[0][0]+XStart+XStep*Member,0,0,0,VyGuess,0]
    x0 = np.concatenate((State0, STM0),axis=1)
    #call targeter to compute orbit
    for target_crossing in range(1,FinalCrossing+1):
        t,x = target_orbit(target_crossing,x0,t0,tf,mu,BackendBigprop,BackendLittleprop,TolIntegrate,TolStep,TolConverge)
    #save desired percentage of converged orbit
    if Member % (100/PercentToPlot) == 0:
        Lyapunovs.append([t,x])
    #report
    print Member,time.time() - Start
    #update guess for next member in family
    VyGuess = x[0][4]

#find rough bifurcation locations
if FinalCrossing == 2:
    Index = 0
    Eigenstructure = []
    BifurcationIndices = []
    for Member in Lyapunovs:
        M = np.reshape(Member[1][-1][6:43],(6,6)) #monodromy matrix is STM after 1 rev
        Eigval,Eigvec = np.linalg.eig(M)
        #get rid of numerical errors in imaginary components of eigenvalues
        Eigval = check_eigenvalues(Eigval,TolEig)
        Eigenstructure.append([Eigval,Eigvec])
        #find orbit indices where stability (number of real eigenvalues) changes
        if Index != 0:
            if np.count_nonzero(np.isreal(Eigenstructure[-1][0])) != np.count_nonzero(np.isreal(Eigenstructure[-2][0])):
                BifurcationIndices.append(Index-1) #save index of inner orbit
        #increment index
        Index += 1

#bisection to find exact bifurcations
for i in range(len(BifurcationIndices)):
    Index = BifurcationIndices[i] + 1*i #because we are injecting the orbits found here into Lyapunovs list, we have to increase the index because the bifurcations will move
    TrajLower = Lyapunovs[Index]
    TrajUpper = Lyapunovs[Index+1]
    EigvalLower = Eigenstructure[Index][0]
    EigvalUpper = Eigenstructure[Index+1][0]
#    Diff = 1
#    while Diff > TolConverge:
    for Counter in range(0,10): #probably close enough after 10 times...but should really use a while loop....
        #target intermediate orbit
        state0 = [(TrajUpper[1][0][0]+TrajLower[1][0][0])/2,0,0,0,(TrajUpper[1][0][4]+TrajLower[1][0][4])/2,0]
        x0 = np.concatenate((state0, STM0),axis=1)
        for target_crossing in range(1,FinalCrossing+1):
            t,x = target_orbit(target_crossing,x0,t0,tf,mu,BackendBigprop,BackendLittleprop,TolIntegrate,TolStep,TolConverge)
        #eigenvalues
        M = np.reshape(x[-1][6:43],(6,6))
        Eigval,Eigvec = np.linalg.eig(M)
        Eigval = check_eigenvalues(Eigval,TolEig)
        #compare eigenvalues, set new upper or lower as appropriate
        if np.count_nonzero(np.isreal(Eigval)) == np.count_nonzero(np.isreal(EigvalLower)):
            TrajLower = [t,x]
            EigvalLower = Eigval
        elif np.count_nonzero(np.isreal(Eigval)) == np.count_nonzero(np.isreal(EigvalUpper)):
            TrajUpper = [t,x]
            EigvalUpper = Eigval
        else:
            sys.exit("Eigenvalues did not match either lower or upper trajectories")
    #insert bifurcation orbit into saved family
    Lyapunovs.insert(Index+1,[t,x])
    Eigenstructure.insert(Index+1,[Eigval,Eigvec])
    #update bifurcation index
    BifurcationIndices[i] = Index+1

#plot
fig = plt.figure()
ax = fig.add_subplot(111)
L1Label, = ax.plot(LibrationPointParams[0][0],LibrationPointParams[1][0],'.') #plot L1
BifurcationIndex = 0
for i in range(len(Lyapunovs)):
    x,y = [],[]
    for j in range(len(Lyapunovs[i][1])):
        x.append(Lyapunovs[i][1][j][0])
        y.append(Lyapunovs[i][1][j][1])    
    LyapunovsLabel, = ax.plot(x,y,'g-')
    if BifurcationIndex < len(BifurcationIndices):
        if i == BifurcationIndices[BifurcationIndex]:
            BifurcationsLabel, = ax.plot(x,y,'k-',linewidth=2)
            BifurcationIndex += 1
    if FinalCrossing == 1:
        MirrorLabel, = ax.plot(x,[-1*k for k in y],'g-')
ax.axis('equal')
fig.show()

#cleanup
del x
del y
print time.time()-Start