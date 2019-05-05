#! /usr/bin/env python
"""
This script implements various iterative methods to
solve the 2D Poisson equation
    
    u_xx + u_yy = f(x,y)

on the square domain x=[0,1] and y=[0,1] with Dirichlet boundary conditions
u(x,0) = u(x,1) = u(0,y) = u(1,y) = 0. For this example,

    f(x,y) = -2*pi^2*sin(pi*x)*sin(pi*y)

corresponding to an exact solution

    u(x,y) = sin(pi*x)*sin(pi*y)

CMSE 821: Numerical Methods for Differential Equations
Michigan State University
Leon Hostetler, Mar. 1, 2018

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


def error_l2(exact, approx):
    """
    This function takes in the exact and approximate solutions ARRAYS
    and returns the error in the l2-norm.
    """
    Soln = exact.copy()
    ASoln = approx.copy()

    vec_Soln = Soln.flatten(order='C')              # Convert exact solution to vector
    vec_U = ASoln.flatten(order='C')                # Convert approx solution to vector
    E = vec_U - vec_Soln                            # Error vector
    err = norm_l2(E, h)                             # l2-norm of error vector

    return err


def jacobi(h, k):
    """
    This function takes in the lattice spacing h and the number of iterations k to perform.
    Applies the steepest descent algorithm and returns the solution array U.

     1) Constructs the guess solution array
     2) Implements the Jacobi algorithm
    """

    m = int(1/h - 1)                                # The number of interior lattice points
                                                    #       in each direction
    U = np.zeros((m+2, m+2), dtype=float)           # Initial solution (approx) array with
                                                    #       boundary conditions

    for k in range(1, k+1):                         # Jacobi method
        new_U = U.copy()
        for j in range(1, m+1):
            for i in range(1, m+1):
                new_U[i,j] = 0.25*((U[i-1,j]+U[i+1,j]+U[i,j-1]+U[i,j+1])-h**2*f(i*h,j*h))
        U = new_U

    return U


def gauss_seidel(h, k):
    """
    This function takes in the lattice spacing h and the number of iterations k to perform.
    Applies the Gauss-Seidel algorithm and returns the solution array U.

     1) Constructs the guess solution array
     2) Implements the Gauss-Seidel algorithm
    """

    m = int(1/h - 1)                                # The number of interior lattice points
                                                    #       in each direction
    U = np.zeros((m+2, m+2), dtype=float)           # Initial solution (approx) array with
                                                    #       boundary conditions

    for k in range(1, k+1):                         # Gauss-Seidel method
        for j in range(1, m+1):
            for i in range(1, m+1):
                U[i,j] = 0.25*((U[i-1,j]+U[i+1,j]+U[i,j-1]+U[i,j+1])-h**2*f(i*h,j*h))

    return U


def sor(h, k):
    """
    This function takes in the lattice spacing h and the number of iterations k to perform.
    Applies the successive overrelaxation algorithm with the optimal parameter for the
    Poisson problem and returns the solution array U.

     1) Constructs the guess solution array
     2) Implements the successive overrelaxation algorithm
    """

    m = int(1/h - 1)                                # The number of interior lattice points
                                                    #       in each direction
    omega = 2.0/(1.0 + np.sin(np.pi*h))             # SOR parameter

    U = np.zeros((m+2, m+2), dtype=float)           # Initial solution (approx) array with
                                                    #       boundary conditions

    for k in range(1, k+1):                                 # SOR method
        for j in range(1, m+1):
            for i in range(1, m+1):
                temp = 0.25*((U[i-1,j]+U[i+1,j]+U[i,j-1]+U[i,j+1])-h**2*f(i*h,j*h))
                U[i,j] = U[i,j] + omega*(temp - U[i,j])

    return U


def steepest_descent(h, k):
    """
    This function takes in the lattice spacing h and the number of iterations k to perform.
    Applies the steepest descent algorithm and returns the solution array U.

     1) Constructs the m^2 x m^2 sparse matrix A corresponding to D^2--the finite
        difference approximation of the second derivative.
     2) Constructs the force vector F from the given function f(x). 
     3) Construct a guess solution and a initial residual vector
     4) Implements the steepest descent algorithm
    """
    m = int(1/h - 1)                                # The number of interior points in each direction
    N = m**2                                        # The length of the vectors

    # Construct the sparse matrix A
    I = sp.eye(m)
    fours = -4.0*np.ones(m, dtype='float')
    ones = np.ones(m-1, dtype='float')
    T = sp.diags([ones, fours, ones], [-1, 0, 1])
    S = sp.diags([ones, ones], [-1, 1])
    A = (sp.kron(I,T) + sp.kron(S,I))/h**2
    #print(A.toarray())                             # Uncomment to verify A for reasonable h

    # Construct the force vector F
    x = np.linspace(h, 1.0, num=m, endpoint=False)  # x-domain
    y = np.linspace(h, 1.0, num=m, endpoint=False)  # y-domain
    X, Y = np.meshgrid(x, y)                        # Grid of x,y values
    Fxy = f(X,Y)                                    # f(x,y) evaluated on the grid
    F = Fxy.flatten(order='C')                      # Flatten it to a vector

    # Steepest Descent
    U = np.zeros(N, dtype=float)                    # Initial guess solution vector
    r = F                                           # Initial residual vector
    for k in range(1, k+1):
        w = A.dot(r)
        a = np.inner(r,r)/np.inner(r,w)
        U = U + a*r
        r = r - a*w

    U = U.reshape((m, m))                           # Unflatten solution vector (convert to array)
    LR = np.zeros(m, dtype='float')
    U = np.hstack((LR[:,None],U,LR[:,None]))        # Reinsert left and right boundary solutions
    TB = np.zeros(m+2, dtype='float')
    U = np.vstack((TB[None,:],U,TB[None,:]))        # Reinsert top and bottom boundary solutions

    return U


def plot_solution(X, Y, Soln, title):
    """
    This function takes in the meshgrid formed by X and Y, the solution array
    Soln, and the title of the plot.
    """
    fig = plt.figure()
    plt.rc('text', usetex=True)
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, U, 200, cmap='viridis')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$u(x,y)$");
    plt.title(title)
    plt.show()


###############################################################
#                        MAIN PART
###############################################################

h = 0.1                                                 # Set the lattice spacing

m = int(1/h - 1)                                        # The number of interior points in each direction

# Exact solution for comparison
xdom = np.linspace(0.0, 1.0, num=m+2, endpoint=True)    # x-domain
ydom = np.linspace(0.0, 1.0, num=m+2, endpoint=True)    # y-domain
X, Y = np.meshgrid(xdom, ydom)                          # Grid of x,y values
Soln = u(X,Y)                                           # The exact solution array


# To approximate the solution using a single method use something like:
#U = jacobi(h, 10)
#U = gauss_seidel(h, 10)
#U = sor(h, 10)
#U = steepest_descent(h, 1)

# Followed by something like
#plot_solution(X, Y, U, "Steepest Descent Method with h = 0.05, k = 1")


K_list = []
err_jac = []
err_gau = []
err_sor = []
err_sdm = []


for K in range(1,10):
    U = jacobi(h, K)
    err_jac.append(error_l2(Soln, U))
    U = gauss_seidel(h, K)
    err_gau.append(error_l2(Soln, U))
    U = sor(h, K)
    err_sor.append(error_l2(Soln, U))
    U = steepest_descent(h, K)
    err_sdm.append(error_l2(Soln, U))
    K_list.append(K)


# Plot the errors vs. k
plt.rc('text', usetex=True)
plt.semilogy(K_list, err_jac, label="Jacobi", marker="|")
plt.semilogy(K_list, err_gau, label="Gauss-Seidel", marker="o", linestyle='dashed')
plt.semilogy(K_list, err_sor, label="SOR", marker="*", linestyle='dotted')
plt.semilogy(K_list, err_sdm, label="Steepest Descent")
plt.legend(loc=1)
plt.xlabel("iterations")
plt.ylabel(r"$||E||_{\ell^2}$")
plt.title(r"$\ell^2$-Errors of Iterative Solvers with $h = 0.1$")
#plt.savefig("errors.png")
plt.show()
