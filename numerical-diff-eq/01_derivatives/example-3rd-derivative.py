#! /usr/bin/env python
"""
This script compares four different finite difference (FD) approximations of the third
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
    """ Define the actual third derivative of fun(x) for comparison"""
    return np.cosh(x)

def third_D(x,eps):
    """ Define the third derivative FD approximation of fun(x)"""
    return (fun(x+2*eps)-3*fun(x+eps)+3*fun(x)-fun(x-eps))/(eps*eps*eps)


# Set the following two lines
x = 1.0                                         # The point to evaluate the derivative at
hlist = [0.1, 0.01, 0.001]                      # The list of h-values to test at

D = [third_D(x,h) for h in hlist]               # List of values third derivative FD approx
De = [D[i]-der(x) for i in range(len(D))]       # Errors of third derivative FD approx.

print("\nThird der. approx. of u(x) with h = 0.001: ",third_D(x,0.001))

# Print the table
print("h       D^3 u(x)-u'''(x)")
for i in range(len(hlist)):
    print(hlist[i],"\t","{:.3e}".format(De[i]))

# Get the slopes
# In general, for $E = |c| h^p, the exponent p can be obtained from 
# the slope of the log-log plot or via the equation
# p = (log(E2)-log(E1))/(log(h2)-log(h1))
# Below, I use this equation to determine p for all four FD methods

logE2, logE1 = np.log10(np.abs(De[len(De)-1])), np.log10(np.abs(De[len(De)-2]))
logh2, logh1 = np.log10(np.abs(hlist[len(hlist)-1])), np.log10(np.abs(hlist[len(hlist)-2]))
p = (logE2-logE1)/(logh2-logh1)
print("\nFor D^3: p = ", p)



# Plot the results
plt.rc('text', usetex=True)
plt.loglog(hlist, np.abs(De), label=r"$D^3 u(\overline{x}) - u'''(\overline{x})$")
plt.legend(loc=2)
plt.xlabel(r"$h$")
plt.ylabel(r"$E_D(h)$")
plt.title("Third Derivative Error at Different h")
#plt.savefig("plot.png")
plt.show()
