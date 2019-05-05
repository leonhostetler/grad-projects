#! /usr/bin/env python
"""

This script implements the streamfunction-vorticity formulation of the Navier Stokes
equations in 2D to solve a lid-driven cavity problem. The cavity shape can be selected
from three options---square, inverted L-shape, and triangular.

NOTE: Typically, I use the index 'i' to refer to the
x-coordinate and the index 'j' refers to the y-coordinate. Then the translation
between Cartesian lattice points and array indices is something like
    A[x,y] := A[j][i]

To use, run this script with the Unix command

    python Hostetler_CMSE821_FINAL_problem3.py

Adjust the FDM parameters (mainly h and the True/False plot choices) as desired and
run again. Note, an h value smaller than 0.02 will take hours to complete.

Note, I am not sure that I implemented the boundary conditions correctly especially for the triangular cavity. Obviously, this code is presented without any kind of implied warranty of its accuracy or stability.

CMSE 821: Numerical Methods for Differential Equations
Michigan State University
Leon Hostetler, May. 3, 2019

"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np


###############################################################
#                    FUNCTION DEFINITIONS
###############################################################

def AB1_timestep(w, p, nu, h, k, cavity_style):
    """
    Solves the vorticity equation for the next time step, using the Adams-Bashforth
    one-step method.

    Takes in:
        w: (matrix) The vorticity solution at the previous time step (aka omega^n)
        p: (matrix) The stream function solution at the previous time step (aka psi^n)
        nu: (number) Kinematic viscosity
        h: (number) The spatial spacing
        k: (number) The temporal spacing
        cavity_style: (number) Flag for cavity shape
    Returns: 
        W: (matrix) The vorticity solution at the current time step (aka omega^(n+1))
    """
    m = int(1/h - 1)
    W = np.zeros((m+2, m+2), dtype=float)
    
    for i in range(1, m+1):
        for j in range(1, m+1):
            A = (p[j+1][i]-p[j-1][i])*(w[j][i+1]-w[j][i-1])
            B = (p[j][i+1]-p[j][i-1])*(w[j+1][i]-w[j-1][i])
            C = w[j][i+1]+w[j+1][i]+w[j][i-1]+w[j-1][i]-4*w[j][i]
            f = -0.25*A/h**2 + 0.25*B/h**2 + nu*C/h**2
           
            W[j][i] = w[j][i] + k*f

    # Boundary values
    C = -2/h**2
    if cavity_style==0: # Square cavity
        for j in range(0, m+2):
            W[j][0] = C*p[j][1]
            W[j][m+1] = C*p[j][m]
        for i in range(0, m+2):
            W[0][i] = C*p[1][i]
            W[m+1][i] = C*p[m][i] - 2/h

    if cavity_style==1: # L-shape cavity
        mid = int((m+1)/2)
        for j in range(0, mid+1):
            W[j][0] = C*p[j][1]         # Bottom left wall
            W[j][mid+1] = C*p[j][mid]   # Bottom right wall
        for j in range(mid, m+1):
            W[j][0] = C*p[j][1]         # Top left wall
            W[j][m+1] = C*p[j][m]       # Top right wall
        for i in range(0, mid+1):
            W[0][i] = C*p[1][i]         # Bottom left
            W[m+1][i] = C*p[m][i] - 2/h # Top
        for i in range(mid, m+1):
            W[mid][i] = C*p[mid+1][i]   # Bottom right
            W[m+1][i] = C*p[m][i] - 2/h # Top

    if cavity_style==2: # Triangular cavity
        mid = int((m+1)/2)
        for j in range(0, m+2):
            W[j][0] = C*p[j][1]         # Left wall
            W[m+1][j] = C*p[m][j] - 2/h # Top
        for j in range(1, m+1):          # Diagonal
            W[j][j] = C*(p[j+1][j] + p[j][j-1])/2.0

    return W


def IMEX_timestep(w0, w1, p0, p1, nu, h, k, cavity_style):
    """
    Solves the vorticity equation for the next time step, using the Adams-Bashforth
    two-step method.

    Takes in:
        w0: (matrix) The vorticity solution at the previous previous time step (aka omega^(n-1))
        p0: (matrix) The stream function solution at the previous previous time step (aka psi^(n-1))
        w1: (matrix) The vorticity solution at the previous time step (aka omega^n)
        p1: (matrix) The stream function solution at the previous time step (aka psi^n)
        nu: (number) Kinematic viscosity
        h: (number) The spatial spacing
        k: (number) The temporal spacing
        cavity_style: (number) Flag for cavity shape
    Returns: 
        W: (matrix) The vorticity solution at the current time step (aka omega^(n+1))
    """
    m = int(1/h - 1)
    W = np.zeros((m+2, m+2), dtype=float)
    P = np.zeros((m+2, m+2), dtype=float)

    P = AB1_timestep(w1, p1, nu, h, k, cavity_style) # AB-1 prediction for AM-step
    
    for i in range(1, m+1):
        for j in range(1, m+1):
            A0 = -(p0[j+1][i]-p0[j-1][i])*(w0[j][i+1]-w0[j][i-1])/(4*h**2)
            B0 = (p0[j][i+1]-p0[j][i-1])*(w0[j+1][i]-w0[j-1][i])/(4*h**2)
            A1 = -(p1[j+1][i]-p1[j-1][i])*(w1[j][i+1]-w1[j][i-1])/(4*h**2)
            B1 = (p1[j][i+1]-p1[j][i-1])*(w1[j+1][i]-w1[j-1][i])/(4*h**2)
            fN1 = A1+B1
            fN0 = A0+B0
            explicit = 3*fN1 - fN0

            fL0 = nu*(w0[j][i+1]+w0[j+1][i]+w0[j][i-1]+w0[j-1][i]-4*w0[j][i])/h**2
            fL1 = nu*(w1[j][i+1]+w1[j+1][i]+w1[j][i-1]+w1[j-1][i]-4*w1[j][i])/h**2
            fL2 = nu*(P[j][i+1]+P[j+1][i]+P[j][i-1]+P[j-1][i]-4*P[j][i])/h**2
            implicit = -1.0*fL0 + 8.0*fL1 + 5.0*fL2

            W[j][i] = w1[j][i] + k*explicit/2.0 + k*implicit/12.0

    # Boundary values
    C = -2/h**2
    if cavity_style==0: # Square cavity
        for j in range(0, m+2):
            W[j][0] = C*p1[j][1]
            W[j][m+1] = C*p1[j][m]
        for i in range(0, m+2):
            W[0][i] = C*p1[1][i]
            W[m+1][i] = C*p1[m][i] - 2/h

    if cavity_style==1: # L-shaped cavity
        mid = int((m+1)/2)
        for j in range(0, mid+1):
            W[j][0] = C*p1[j][1]         # Bottom left wall
            W[j][mid+1] = C*p1[j][mid]   # Bottom right wall
        for j in range(mid, m+1):
            W[j][0] = C*p1[j][1]         # Top left wall
            W[j][m+1] = C*p1[j][m]       # Top right wall
        for i in range(0, mid+1):
            W[0][i] = C*p1[1][i]         # Bottom left
            W[m+1][i] = C*p1[m][i] - 2/h # Top
        for i in range(mid, m+1):
            W[mid][i] = C*p1[mid+1][i]   # Bottom right
            W[m+1][i] = C*p1[m][i] - 2/h # Top

    if cavity_style==2: # Triangular cavity
        mid = int((m+1)/2)
        for j in range(0, m+2):
            W[j][0] = C*p1[j][1]         # Left wall
            W[m+1][j] = C*p1[m][j] - 2/h # Top
        for j in range(1, m+1):          # Diagonal
            W[j][j] = C*(p1[j+1][j] + p1[j][j-1])/2.0

    return W


def solve_poisson(w, h, tol, maxit, method, cavity_style):
    """
    Applies the Gauss-Seidel algorithm and returns the solution array U.

    Takes in:
        w: (matrix) The vorticity solution at the current time step
        h: (number) The spatial spacing
        tol: (number) The tolerance for the norm of the difference between successive iterations
        maxit: (number) The maximum number of iterations to perform
        method: (number) 0 for Gauss-Seidel, 1 for SOR
        cavity_style: (number) Flag for cavity shape

    Returns:
        itnum: The number of iterations used to solve the Poisson equation
        P: The stream function solution at the current time
    """
    m = int(1/h - 1)                        # Interior lattice points in each direction
    P = np.zeros((m+2, m+2), dtype=float)   # Solution array
    itnum = 0                               # Iteration number
    norm = 1.0                              # Initialize to arbitrary (large) value

    omega = 2.0/(1.0 + np.sin(np.pi*h)) # SOR parameter
    while norm > tol and itnum < maxit:
        P_old = np.copy(P)
        for j in range(1, m+1):
            for i in range(1, m+1):
                temp = 0.25*(P[j][i-1] + P[j][i+1] + P[j-1][i] + P[j+1][i] + h**2*w[j][i])
                if method == 0: # Gauss-Seidel
                    P[j][i] = temp
                if method == 1: # SOR step
                    P[j][i] = P[j][i] + omega*(temp - P[j][i])

                if cavity_style==1: # Cut out the bottom-right corner
                    if (i*h>=0.5) and (j*h<=0.5):
                        P[j][i] = 0.0
                if cavity_style==2: # Cut out the bottom-right
                    if (j/i<=1.0):
                        P[j][i] = 0.0

        diff = np.subtract(P_old, P)
        norm = np.linalg.norm(diff)
        itnum += 1

    return itnum, P


###############################################################
#                        MAIN PART
###############################################################

# FDM parameters and plotting choices
h = 0.05                     # Spatial spacing
nu = 0.01                    # Kinematic viscosity
k = 0.2*h**2/nu              # Temporal spacing
tol = 1e-5                   # Difference tolerance for Poisson solver, Rec: 1e-5
maxit = 1000                 # Max iterations for Gauss-Seidel
t_tol = 1e-7                 # Steady-state equilibration tolerance, Rec: 1e-7
poisson_method = 1           # 0 for Gauss-Seidel, 1 for SOR
shape = 1                    # Cavity shape: 0 (square), 1 (inverted L), 2 (triangle)
plot_velfld = False          # Do you want to plot the velocity field?
plot_strlns = True           # Do you want to plot the stream lines?
plot_strfun = False          # Do you want to plot the stream function?
plot_vortic = False          # Do you want to plot vorticity?

print("FDM Parameters")
print("--------------------------------")
print("Lattice size X x Y: ", 1/h, " x ", 1/h)
print("h = ", h, ", k = ", k)
print("nu = ", nu, ", Re = ", 1/nu)

# Initialize
m = int(1/h - 1)             # The number of interior points in each direction
W0 = np.zeros((m+2, m+2), dtype=float)
P0 = np.zeros((m+2, m+2), dtype=float)         
time, t_steps, itsum = 0.0, 0, 0.0
v_avg, v_diff = 1.0, 1.0

# Initial time step
W1 = AB1_timestep(W0, P0, nu, h, k, shape)
its, P1 = solve_poisson(W1, h, tol, maxit, poisson_method, shape)

# Loop until equilibrated
while v_diff > t_tol:
    v_old = v_avg
    W2 = IMEX_timestep(W0, W1, P0, P1, nu, h, k, shape)
    its, P2 = solve_poisson(W2, h, tol, maxit, poisson_method, shape)

    # Check for steady-state equilibration
    vel = []
    for i in range(1, m+1):
        if shape==0 or shape==1:
            # Sample near bottom
            vel += [-(P2[0][i]-P2[2][i])/(2*h)]
        elif shape==2:
            # Sample near middle
            mid = int((m+1)/2)
            vel += [-(P2[i][mid]-P2[i][mid+2])/(2*h)]
    v_avg = np.average(vel)
    v_diff = abs(v_avg - v_old)

    W0, W1, P0, P1 = W1, W2, P1, P2
    time += k
    t_steps += 1
    itsum += its

print("Average Poisson eqn iterations: ", itsum/t_steps)
print("Final time: ", t_steps*k)

psi = P1
vor = W1

# Velocity
u = np.zeros((m+2, m+2), dtype=float) # x-components
v = np.zeros((m+2, m+2), dtype=float) # y-components
for i in range(1, m+1):
    for j in range(1, m+1):
        u[j][i] = (psi[j+1][i]-psi[j-1][i])/(2*h)
        v[j][i] = -(psi[j][i+1]-psi[j][i-1])/(2*h)
for i in range(0, m+2):
    u[m+1][i] = 1.0

# Plot grid
xdom = np.linspace(0.0, 1.0, num=m+2, endpoint=True)    # x-domain
ydom = np.linspace(0.0, 1.0, num=m+2, endpoint=True)    # y-domain
X, Y = np.meshgrid(xdom, ydom)                          # Grid of x,y values

if plot_velfld: # Plot velocity fields
    zeros = np.zeros((m+2, m+2), dtype=float)

    plt.quiver(X, Y, u, v, alpha=.5)
    plt.quiver(X, Y, u, v, edgecolor='k', facecolor='None', linewidth=.5)
    plt.show()

    plt.quiver(X, Y, u, zeros, alpha=.5) # x-components only
    plt.quiver(X, Y, u, zeros, edgecolor='k', facecolor='None', linewidth=.5)
    plt.show()

    plt.quiver(X, Y, zeros, v, alpha=.5) # y-components only
    plt.quiver(X, Y, zeros, v, edgecolor='k', facecolor='None', linewidth=.5)
    plt.show()


if plot_strlns: # Plot stream lines
    fig, ax0 = plt.subplots()
    strm = ax0.streamplot(X, Y, u, v, color=u, linewidth=1, cmap='cool')
    fig.colorbar(strm.lines)
    ax0.set_title('Stream Lines')
    plt.show()


if plot_strfun: # Plot stream function
    fig1, ax = plt.subplots()
    im = ax.imshow(psi, interpolation='bilinear', cmap='cool', origin='lower')
    ax.set_title('Stream Function')
    plt.show()


if plot_vortic: # Plot vorticity
    fig1, ax = plt.subplots()
    im = ax.imshow(vor, interpolation='bilinear', cmap='cool', origin='lower')
    ax.set_title('Vorticity')
    plt.show()
