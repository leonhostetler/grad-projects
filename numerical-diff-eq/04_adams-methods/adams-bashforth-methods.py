#! /usr/bin/env python
"""
This script implements the linear multi-step Adams-Bashforth method to solve
the ODE u'(t) = -u. It uses the 1-, 2-, and 3-step methods.

CMSE 821: Numerical Methods for Differential Equations
Michigan State University
Leon Hostetler, Apr. 2, 2018

"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

###############################################################
#                    FUNCTION DEFINITIONS
###############################################################

def u(t):
    """
        This is the exact function, which is known in this
        case, so we want to compare it with the FDM solution.
    """
    return np.exp(-t)


def AB1(start, stop, k, ival):
    """
    The Adams-Bashforth 1-step algorithm for
        u'(t) = -u

    Takes in
        start: The start of the time domain
        stop: The end of the time domain
        k: Length of the time steps
        ival: The initial value U(start)

    Returns
        t: The vector of time values
        fdm: The solution vector
    """
    N = (stop-start)/k                              # Number of points
    t = np.linspace(start,stop,num=N+1,dtype=float) # Vector of time values
    fdm = []                                        # Solution vector

    fdm.append(ival)                                # Add the initial value
    for i in range(int(N)):                         # Use AB1 steps for the rest of the steps
        fdm.append((1.0-k)*fdm[i])

    return t, fdm


def AB2(start, stop, k, ival):
    """
    The Adams-Bashforth 2-step algorithm for
        u'(t) = -u

    Takes in
        start: The start of the time domain
        stop: The end of the time domain
        k: Length of the time steps
        ival: The initial value U(start)

    Returns
        t: The vector of time values
        fdm: The solution vector
    """
    N = (stop-start)/k                              # Number of points
    t = np.linspace(start,stop,num=N+1,dtype=float) # Vector of time values
    fdm = []                                        # Solution vector

    fdm.append(ival)                                # Add the initial value
    fdm.append((1.0-k)*fdm[0])                      # Add the second value using an AB1 step
    for i in range(int(N-1)):                       # Use AB2 steps for the rest of the steps
        fdm.append((1.0-1.5*k)*fdm[i+1] +0.5*k*fdm[i])

    return t, fdm


def AB3(start, stop, k, ival):
    """
    The Adams-Bashforth 3-step algorithm for
        u'(t) = -u

    Takes in
        start: The start of the time domain
        stop: The end of the time domain
        k: Length of the time steps
        ival: The initial value U(start)

    Returns
        t: The vector of time values
        fdm: The solution vector
    """
    N = (stop-start)/k                              # Number of points
    t = np.linspace(start,stop,num=N+1,dtype=float) # Vector of time values
    fdm = []                                        # Solution vector

    fdm.append(ival)                                # Add the initial value
    fdm.append((1.0-k)*fdm[0])                      # Add the second value using an AB1 step

    # Somehow, using an AB2 step for the third value destroys the order 3 accuracy
    # of this method. Try it by toggling the test flag True/False
    if test:
        fdm.append(u(2*k))                          # Add the third value exactly (i.e. cheating)
    else:
        fdm.append(fdm[1]+0.5*k*(fdm[0]-3*fdm[1]))  # Add the third value using an AB2 step

    for i in range(3,int(N+1)):                     # Use AB3 for the rest of the steps
        fdm.append(fdm[i-1] + k*(-5.0*fdm[i-3] + 16.0*fdm[i-2]-23.0*fdm[i-1])/12.0)

    return t, fdm


###############################################################
#                        MAIN PART
###############################################################

klist = [0.1, 0.01, 0.001, 0.001, 0.0001]    # The list of k-values to test at
tlist1, tlist2, tlist3 = [], [], []          # The lists of domain vectors
Ulist1, Ulist2, Ulist3 = [], [], []          # The lists of solution vectors
err1, err2, err3 = [], [], []

tstart = 0.0
tstop = 1.0
initial_value = 1.0     # If you change this, you must also redefine the function u(t)
test = True             # Test flag

# Get the solutions (for the different k-values)
for i in range(len(klist)):
    t1, U1 = AB1(tstart, tstop, klist[i], initial_value)
    t2, U2 = AB2(tstart, tstop, klist[i], initial_value)
    t3, U3 = AB3(tstart, tstop, klist[i], initial_value)
    tlist1.append(t1)
    tlist2.append(t2)
    tlist3.append(t3)
    Ulist1.append(U1)
    Ulist2.append(U2)
    Ulist3.append(U3)


# Calculate the error in the last elements
for i in range(len(klist)):
    err1.append(np.abs(Ulist1[i][-1] - u(tstop)))
    err2.append(np.abs(Ulist2[i][-1] - u(tstop)))
    err3.append(np.abs(Ulist3[i][-1] - u(tstop)))


# Get the slopes
logE2, logE1 = np.log10(np.abs(err1[-1])), np.log10(np.abs(err1[1]))
logh2, logh1 = np.log10(np.abs(klist[-1])), np.log10(np.abs(klist[1]))
print("\nFor AB1: p = ", (logE2-logE1)/(logh2-logh1))

logE2, logE1 = np.log10(np.abs(err2[-1])), np.log10(np.abs(err2[1]))
print("\nFor AB2: p = ", (logE2-logE1)/(logh2-logh1))

logE2, logE1 = np.log10(np.abs(err3[-1])), np.log10(np.abs(err3[1]))
print("\nFor AB3: p = ", (logE2-logE1)/(logh2-logh1))


# Plot the errors vs. k
plt.rc('text', usetex=True)
plt.loglog(klist, err1, label="AB1", marker='*')
plt.loglog(klist, err2, label="AB2", marker='*')
plt.loglog(klist, err3, label="AB3", marker='*')
plt.legend(loc=2)
plt.xlabel(r"$k$")
plt.ylabel(r"$E$")
plt.title("Error of Adams-Bashforth Methods")
plt.show()


# If you want to plot one of the FDM approximations and the exact solution
# for comparison, then set plotFDM to 'True' and select an index i
# corresponding to the desired k-value. For example, i=0 means use the first
# k-value in klist
plotFDM = False
i = 0
if plotFDM:
    tval = np.linspace(tstart, tstop, 1000)  # x-values for exact function
    plt.rc('text', usetex=True)
    plt.plot(tlist1[i], Ulist1[i], label="AB1", marker ='+')
    plt.plot(tlist2[i], Ulist2[i], label="AB2", marker ='+')
    plt.plot(tlist3[i], Ulist3[i], label="AB3", marker ='+')
    plt.plot(tval, u(tval), label="Exact solution", linestyle='dashed')
    plt.legend(loc=1)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u(t), \,U_j$")
    plt.title("Comparison of Adams-Bashforth vs. Exact Solution")
    plt.show()
