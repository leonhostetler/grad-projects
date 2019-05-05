#! /usr/bin/env python
"""
This script implements a finite difference approximation for the solution u(x) of
the differential equation u'' = f(x), with boundary conditions u(0) = u(1) = 0. In
this case, we are given that f(x) = -pi^2*sin(pi*x). The exact solution, to be used
for comparison, is u(x) = sin(pi*x).

Here, we use a finite difference method (FDM) to solve the ODE. We use the FD
approximation

    (U_{j-1} - 2U_j + U_{j+1})/h^2 ~ f_j

at all m interior points of our domain. This gives us a set of linear equations,
which can be written as the matrix equation AU = F, where A is an m x m tridiagonal
and symmetric matrix. Then our FDM solution is U = A^{-1}F.

To use:
    1) Set your choice of lattice spacing h
    2) Run the script

CMSE 821: Numerical Methods for Differential Equations
Michigan State University
Leon Hostetler, Feb. 7, 2018

"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """
        This function returns the f(x) of u''=f(x).
        In our case, it is f(x) = - pi^2*sin(pi*x)
    """
    return -np.pi*np.pi*np.sin(np.pi*x)


def u(x):
    """
        This is the exact function, which is known in this
        case, so we want to compare it with the FDM solution.
    """
    return np.sin(np.pi*x)


def norm_l1(vec, eps):
    """
        Takes a vector vec and the lattice spacing eps and returns
        the l1-norm of the given vector
    """
    return eps*np.sum(np.absolute(vec))


def norm_l2(vec, eps):
    """
        Takes a vector vec and the lattice spacing eps and returns
        the l2-norm of the given vector
    """
    return np.sqrt(eps*np.sum(np.absolute(vec)**2))


def norm_inf(approx, exact):
    """
        Takes two vectors---an approximation and an exact and
        returns the infinity-norm.
    """
    return np.amax(np.absolute(approx-exact))



h = 0.25             # Lattice spacing
m = int(1/h - 1)    # The number of interior lattice points
left = 0.0          # Left endpoint of domain
right = 1.0         # Right endpoint of domain

print("Lattice spacing: ", h)
print("Interior points: ", m)
print("Domain: [",left,",",right,"]")

"""
Construct the m x m matrix A corresponding to D^2--the finite difference
approximation of the second derivative. This matrix should have -2.0 on
the diagonal, and 1.0 on both the super- and subdiagonals.
"""
A = np.zeros((m, m), dtype=float)   # Declare the matrix and fill with zeros

for i in range(m):                  # Add the elements on the tridiagonal
    for j in range(m):
        if i==j:
            A[i][j] = -2.0
        if i==j+1 or i ==j-1:
            A[i][j] = 1.0

print("\nMatrix A (before scaled by 1/h^2):\n", A)

A = np.divide(A, h*h)               # Divide A by h^2 to properly scale it

"""
Construct the vector of x-values. This is an m x 1 vector of equally-spaced
values between (not including) the endpoints of the domain. The spacing
is given by h.
"""
x = np.linspace(left+h, right, num=m, endpoint=False, dtype=float)

print("\nx-values of interior points:\n", x)


"""
Construct the force vector F. In general, this includes the known values
at the endpoints, but in this example, u(0) = u(1) = 0.
"""
F = f(x)

print("\nForce vector:\n", F)


"""
Obtain the FDM solution vector U = A^(-1)*F
"""
U = np.linalg.inv(A).dot(F)

print("\nFDM Solution:\n", U)


"""
Calculate the errors. The error vector is E = U_j - u(x_j), where U_j is the
FDM solution, and u(x_j) is the exact solution evaluated at the same x-values.
Note, I am not including the endpoints in the error vector, but that is fine. 
The endpoints are known, so the errors are zero there.
"""
E = U-u(x)                                          # The error vector

print("\nError vector: \n", E)
print("Infinity-norm: ", norm_inf(U,u(x)))
print("l1-norm: ", norm_l1(E,h))
print("l2-norm: ", norm_l2(E,h))

"""
Plot the FDM solution and the exact solution for comparison
"""
xval = np.linspace(left, right, 1000)  # x-values for exact function

# Add the known endpoints to the approximate solution
x = np.hstack((np.array((left)),x,np.array((right))))
U = np.hstack((np.array((0.0)),U,np.array((0.0))))
E = np.hstack((np.array((0.0)),E,np.array((0.0))))

plt.rc('text', usetex=True)
plt.plot(x, U, label="FDM solution", marker='+')
plt.plot(xval, u(xval), label="Exact solution", linestyle='dashed')
plt.plot(x, E, label="Error", marker='4')
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.legend(loc=1)
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x), \,U_j$")
plt.title("Comparison of FDM vs. Exact Solution")
plt.show()

