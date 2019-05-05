#! /usr/bin/env python
"""
This script compares four different finite difference (FD) approximations of the first
derivative of a given function---the forward, backward, centered, and a third-order
approximation. For each FD method, this script prints out the error for each value of h, prints out the computational order of convergence p, and plots the error.

CMSE 821: Numerical Methods for Differential Equations
Michigan State University
Leon Hostetler, Jan. 15, 2018

To use:
    1) Define a smooth function whose derivative you want approximated
       at some point in fun(x)
    2) Define its "exact" derivative for later comparison against the FD approx.
       in der(x)
    3) Set the variable x, where you want the derivative approximated
    4) Add your desired values of h to test in hlist
    5) Run the code with the command:

            python hostetler_hw1_1.py

"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np


def fun(x):
    """ Define the smooth function whose derivative is to be approximated"""
    return np.sinh(x)

def der(x):
    """ Define the actual derivative of fun(x) for comparison"""
    return np.cosh(x)

def D_plus(x,eps):
    """ Define the forward finite difference approximation of the derivative of fun(x)"""
    return (fun(x+eps)-fun(x))/eps

def D_minus(x,eps):
    """ Define the forward finite difference approximation of the derivative of fun(x)"""
    return (fun(x)-fun(x-eps))/eps

def D_zero(x,eps):
    """ Define the forward finite difference approximation of the derivative of fun(x)"""
    return (fun(x+eps)-fun(x-eps))/(2*eps)

def D_three(x,eps):
    """ Define the forward finite difference approximation of the derivative of fun(x)"""
    return (2*fun(x+eps)+3*fun(x)-6*fun(x-eps)+fun(x-2*eps))/(6*eps)


# Set the following two lines
x = 1.0                                         # The point to evaluate the derivative at
hlist = [0.1, 0.01, 0.001]                      # The list of h-values to test at

Dp = [D_plus(x,h) for h in hlist]               # List of values from forward FD approx
Dm = [D_minus(x,h) for h in hlist]              # List of values from backward FD approx
Dz = [D_zero(x,h) for h in hlist]               # List of values from centered FD approx
D3 = [D_three(x,h) for h in hlist]              # List of values from third-order FD approx

Dpe = [Dp[i]-der(x) for i in range(len(Dp))]    # Errors of forward FD approx.
Dme = [Dm[i]-der(x) for i in range(len(Dm))]    # Errors of backward FD approx.
Dze = [Dz[i]-der(x) for i in range(len(Dz))]    # Errors of centered FD approx.
D3e = [D3[i]-der(x) for i in range(len(D3))]    # Errors of third-order FD approx.

# Print the table
print("h       D_+u(x)-u'(x)   D_-u(x)-u'(x)   D_0u(x)-u'(x)   D_3u(x)-u'(x)")
for i in range(len(hlist)):
    print(hlist[i],"\t","{:.3e}".format(Dpe[i]),"\t","{:.3e}".format(Dme[i]),"\t","{:.3e}".format(Dze[i]),"\t","{:.3e}".format(D3e[i]))

# Get the slopes
# In general, for $E = |c| h^p, the exponent p can be obtained from 
# the slope of the log-log plot or via the equation
# p = (log(E2)-log(E1))/(log(h2)-log(h1))
# Below, I use this equation to determine p for all four FD methods

logE2, logE1 = np.log10(np.abs(Dpe[2])), np.log10(np.abs(Dpe[0]))
logh2, logh1 = np.log10(np.abs(hlist[2])), np.log10(np.abs(hlist[0]))
p = (logE2-logE1)/(logh2-logh1)
print("\nFor D_+: p = ", p)

logE2, logE1 = np.log10(np.abs(Dme[2])), np.log10(np.abs(Dme[0]))
p = (logE2-logE1)/(logh2-logh1)
print("For D_-: p = ", p)

logE2, logE1 = np.log10(np.abs(Dze[2])), np.log10(np.abs(Dze[0]))
p = (logE2-logE1)/(logh2-logh1)
print("For D_0: p = ", p)

logE2, logE1 = np.log10(np.abs(D3e[2])), np.log10(np.abs(D3e[0]))
p = (logE2-logE1)/(logh2-logh1)
print("For D_3: p = ", p)


# Plot the results
plt.rc('text', usetex=True)
plt.loglog(hlist, np.abs(Dpe), label=r"$D_+u(\overline{x}) - u'(\overline{x})$")
plt.loglog(hlist, np.abs(Dme), label=r"$D_-u(\overline{x}) - u'(\overline{x})$")
plt.loglog(hlist, np.abs(Dze), label=r"$D_0u(\overline{x}) - u'(\overline{x})$")
plt.loglog(hlist, np.abs(D3e), label=r"$D_3u(\overline{x}) - u'(\overline{x})$")
plt.legend(loc=2)
plt.xlabel(r"$h$")
plt.ylabel(r"$E_D(h)$")
plt.title("FD Derivative Algorithms Comparison at Different h")
#plt.savefig("plot.png")
plt.show()
