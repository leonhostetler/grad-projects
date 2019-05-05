#! /usr/bin/env python
"""
This script implements the Gauss-Seidel (iterative) method to solve the 2D Poisson equation
    
    u_xx + u_yy = f(x,y)

on the square domain x=[0,1] and y=[0,1] with Dirichlet boundary conditions
u(x,0) = u(x,1) = u(0,y) = u(1,y) = 0. For this example,

    f(x,y) = -2*pi^2*sin(pi*x)*sin(pi*y)

corresponding to an exact solution

    u(x,y) = sin(pi*x)*sin(pi*y)

CMSE 821: Numerical Methods for Differential Equations
Michigan State University
Leon Hostetler, Feb. 24, 2018

"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

###############################################################
#                    FUNCTION DEFINITIONS
###############################################################

def f(x,y):
    """
        This function returns the f(x,y) of u_xx + u_yy = f(x,y).
        In our case, it is f(x,y) = -2*pi^2*sin(pi*x)*sin(pi*y)
    """
    return -2.0*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)


def u(x,y):
    """
        This is the exact function, which is known in this
        case, so we want to compare it with the iterative solution.
    """
    return np.sin(np.pi*x)*np.sin(np.pi*y)


def norm_l2(vec, eps):
    """
        Takes a vector vec and the lattice spacing eps and returns
        the l2-norm of the given vector
    """
    return np.sqrt(eps*np.sum(np.square(vec)))


###############################################################
#                        MAIN PART
###############################################################

h = 0.1                                                 # The lattice spacing in both x and y directions
k = 1000                                                # The number of iterations


m = int(1/h - 1)                                        # The number of interior lattice points
                                                        #       in each direction
U = np.zeros((m+2, m+2), dtype=float)                   # Initial solution (approx) array with
                                                        #       boundary conditions

for k in range(1, k+1):                                 # Gauss-Seidel method
    for j in range(1, m+1):
        for i in range(1, m+1):
            U[i,j] = 0.25*((U[i-1,j]+U[i+1,j]+U[i,j-1]+U[i,j+1])-h**2*f(i*h,j*h))


xdom = np.linspace(0.0, 1.0, num=m+2, endpoint=True)    # x-domain
ydom = np.linspace(0.0, 1.0, num=m+2, endpoint=True)    # y-domain
X, Y = np.meshgrid(xdom, ydom)                          # Grid of x,y values
Soln = u(X,Y)                                           # The exact solution array

vec_U = U.flatten(order='C')                            # Convert approx solution U to vector
vec_Soln = Soln.flatten(order='C')                      # Convert exact solution Soln to vector
E = vec_U - vec_Soln                                    # Error vector
err = norm_l2(E, h)                                     # l2-norm of error vector

print("l2 error: ", err)

###############################################################
#                        PLOTTING
###############################################################

fig = plt.figure()
plt.rc('text', usetex=True)
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Soln, 100, cmap='viridis')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$u(x,y)$");
plt.title("Gauss-Seidel Method")
plt.show()
