#! /usr/bin/env python
"""
This script implements the steepest descent (iterative) method to
solve the 2D Poisson equation
    
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
from scipy import sparse as sp

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

h = 0.01                                                 # The lattice spacing in both x and y directions
k = 5                                                # The number of iterations

m = int(1/h - 1)                                        # The number of interior points in each direction
N = m**2                                                # The length of the vectors

I = sp.eye(m)                                           # Construct the sparse matrix A
fours = -4.0*np.ones(m, dtype='float')
ones = np.ones(m-1, dtype='float')
T = sp.diags([ones, fours, ones], [-1, 0, 1])
S = sp.diags([ones, ones], [-1, 1])
A = (sp.kron(I,T) + sp.kron(S,I))/h**2
#print(A.toarray())                                     # Uncomment to verify A for reasonable h

# Construct the force vector F
x = np.linspace(h, 1.0, num=m, endpoint=False)          # x-domain
y = np.linspace(h, 1.0, num=m, endpoint=False)          # y-domain
X, Y = np.meshgrid(x, y)                                # Grid of x,y values
Fxy = f(X,Y)                                            # f(x,y) evaluated on the grid
F = Fxy.flatten(order='C')                              # Flatten it to a vector

U = np.zeros(N, dtype=float)                            # Initial guess solution
r = F                                                   # Initial residual vector

for k in range(1, k+1):
    w = A.dot(r)
    a = np.inner(r,r)/np.inner(r,w)
    U = U + a*r
    r = r - a*w


Soln = u(X,Y)                                           # The exact solution array
vec_Soln = Soln.flatten(order='C')                      # Convert exact solution Soln to vector
E = U - vec_Soln                                        # Error vector
err = norm_l2(E, h)                                     # l2-norm of error vector

print("l2 error: ", err)

U = U.reshape((m, m))                                   # Unflatten the solution vector


###############################################################
#                        PLOTTING
###############################################################

fig = plt.figure()
plt.rc('text', usetex=True)
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, U, 200, cmap='viridis')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$u(x,y)$");
plt.title("Steepest Descent Method")
plt.show()
