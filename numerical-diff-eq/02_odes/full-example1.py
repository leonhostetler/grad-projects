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

CMSE 821: Numerical Methods for Differential Equations
Michigan State University
Leon Hostetler, Feb. 7, 2018

"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

###############################################################
#                    FUNCTION DEFINITIONS
###############################################################

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
    return np.sqrt(eps*np.sum(np.square(vec)))


def norm_inf(approx, exact):
    """
        Takes two vectors---an approximation and an exact and
        returns the infinity-norm.
    """
    return np.amax(np.absolute(approx-exact))


def fdm_interior_solve(left, right, h):
    """
    This function takes in the left and right endpoints of the domain (e.g. [0,1])
    and the lattice spacing h. Returns the domain vector x and the FDM solution vector U.

     1) Constructs the m x m matrix A corresponding to D^2--the finite
        difference approximation of the second derivative.
     2) Constructs the domain vector x of interior x_j values
     3) Constructs the force vector F from the given function f(x). 
        Generally includes boundary conditions, but in this case, the 
        boundary points are zero.
     4) Obtains the FDM solution U = A^{-1}F. 

    NOTE: The solution vector returned does not contain the endpoints of the
        domain (which are known).
    """
    m = int(1/h - 1)                    # The number of interior lattice points

    A = np.zeros((m, m), dtype=float)   # Declare the matrix A and fill it with zeros
    for i in range(m):                  # Add the elements on the tridiagonal
        for j in range(m):
            if i==j:                    # -2.0 on the diagonal
                A[i][j] = -2.0
            if i==j+1 or i ==j-1:       # +1.0 on the super- and subdiagonals
                A[i][j] = 1.0
    A = np.divide(A, h*h)               # Divide A by h^2 to properly scale it

    x = np.linspace(left+h, right, num=m, endpoint=False, dtype=float)
    F = f(x)                            # The force vector F
    U = np.linalg.inv(A).dot(F)         # The FDM solution U = A^{-1}F

    return x, U


###############################################################
#                        MAIN PART
###############################################################

hlist = [0.1, 0.05, 0.01, 0.005, 0.001]             # The list of h-values to test at
xlist = []                                          # The list of domain vectors (one for each h)
Ulist = []                                          # The list of solution vectors (one for each h)
inf_list = []                                       # The list of infinity-norms of the errors
l1_list = []                                        # The list of l1-norms of the errors
l2_list = []                                        # The list of l2-norms of the errors

# Get the solutions (for the different h-values)
for i in range(len(hlist)):
    x, U = fdm_interior_solve(0.0, 1.0, hlist[i])
    xlist.append(x)
    Ulist.append(U)

# Calculate the errors
for i in range(len(hlist)):
    E = Ulist[i] - u(xlist[i])                      # The error vector
    inf_list.append(norm_inf(Ulist[i], u(xlist[i])))# The infinity-norm
    l1_list.append(norm_l1(E, hlist[i]))            # The l1-norm
    l2_list.append(norm_l2(E, hlist[i]))            # The l2-norm

# Print the table of errors
print("h        ||E||_inf         ||E||_1        ||E||_2")
for i in range(len(hlist)):
    print(hlist[i],"\t","{:.3e}".format(inf_list[i]),"\t","{:.3e}".format(l1_list[i]),"\t","{:.3e}".format(l2_list[i]))

# Get the slopes
logE2, logE1 = np.log10(np.abs(inf_list[4])), np.log10(np.abs(inf_list[2]))
logh2, logh1 = np.log10(np.abs(hlist[4])), np.log10(np.abs(hlist[2]))
p = (logE2-logE1)/(logh2-logh1)
print("\nFor ||E||_inf: p = ", p)

logE2, logE1 = np.log10(np.abs(l1_list[4])), np.log10(np.abs(l1_list[2]))
p = (logE2-logE1)/(logh2-logh1)
print("\nFor ||E||_1: p = ", p)

logE2, logE1 = np.log10(np.abs(l2_list[4])), np.log10(np.abs(l2_list[2]))
p = (logE2-logE1)/(logh2-logh1)
print("\nFor ||E||_2: p = ", p)

# Plot the errors vs. h
plt.rc('text', usetex=True)
plt.loglog(hlist, inf_list, label=r"$||E||_{\infty}$", marker='*')
plt.loglog(hlist, l1_list, label=r"$||E||_1$", marker='+')
plt.loglog(hlist, l2_list, label=r"$||E||_2$", marker='4')
plt.legend(loc=2)
plt.xlabel(r"$h$")
plt.ylabel(r"$||E||_*$")
plt.title("Error Norms")
#plt.savefig("errors.png")
plt.show()


# If you want to plot one of the FDM approximations and the exact solution
# for comparison, then set plotFDM to 'True' and select an index i
# corresponding to the desired h-value. For example, i=0 means use the first
# h-value in hlist
plotFDM = False
i = 0
if plotFDM:
    xval = np.linspace(0.0, 1.0, 1000)  # x-values for exact function

    # Add the known endpoints to the FDM solution
    x = np.hstack((np.array((0.0)),xlist[i],np.array((1.0))))
    U = np.hstack((np.array((0.0)),Ulist[i],np.array((0.0))))

    plt.rc('text', usetex=True)
    plt.plot(x, U, label="FDM solution", marker='+')
    plt.plot(xval, u(xval), label="Exact solution", linestyle='dashed')
    plt.legend(loc=2)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x), \,U_j$")
    plt.title("Comparison of FDM vs. Exact Solutions")
    plt.show()

