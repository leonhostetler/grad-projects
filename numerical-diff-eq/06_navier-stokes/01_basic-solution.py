#! /usr/bin/env python
"""
This is a very basic (prototype) solver for the streamfunction-vorticity
formulation of the Navier-Stokes equations in 2D for a lid-driven
rectangular cavity.

CMSE 821: Numerical Methods for Differential Equations
Michigan State University
Leon Hostetler, Apr. 24, 2019

"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np


###############################################################
#                    FUNCTION DEFINITIONS
###############################################################

def init(h):
    """
    Initial conditions
    """
    m = int(1/h - 1)
    w0 = np.zeros((m+2, m+2), dtype=float)
    p0 = np.zeros((m+2, m+2), dtype=float)
    
    return w0, p0


def AB1_timestep(w, p, nu, h, k):
    """
    Solves the vorticity equation for the next time step, using the Adams-Bashforth
    one-step method.

    Takes in:
        w: (matrix) The vorticity solution at the previous time step (aka omega^n)
        p: (matrix) The stream function solution at the previous time step (aka psi^n)
        nu: (number) Kinematic viscosity
        h: (number) The spatial spacing
        k: (number) The temporal spacing
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

    for j in range(0, m+2):
        W[j][0] = C*p[j][1]
        W[j][m+1] = C*p[j][m]

    for i in range(0, m+2):
        W[0][i] = C*p[1][i]
        W[m+1][i] = C*p[m][i] - 2/h
   
    return W


def gauss_seidel(w, h, it):
    """
    Applies the Gauss-Seidel algorithm and returns the solution array U.

    Takes in:
        w: (matrix) The vorticity solution at the current time step
        h: (number) The spatial spacing
        it: (number) The number of iterations to perform

    Returns:
        P: The stream function solution at the current time
    """

    m = int(1/h - 1)                                # The number of interior lattice points
                                                    #       in each direction
    P = np.zeros((m+2, m+2), dtype=float)           # Initial solution (approx) array with
                                                    #       boundary conditions

    for k in range(1, it+1):                        # Gauss-Seidel method
        for j in range(1, m+1):
            for i in range(1, m+1):
                P[j][i] = 0.25*((P[j][i-1]+P[j][i+1]+P[j-1][i]+P[j+1][i])+h**2*w[j][i])

    return P




###############################################################
#                        MAIN PART
###############################################################

h = 0.05                                                 # Spatial spacing
k = 0.8*h                                               # Temporal spacing
nu = 0.01

tf = 20

m = int(1/h - 1)                                        # The number of interior points in each direction

W0, P0 = init(h)

time = 0.0
while time < tf:
#while time < 100*k:
    W1 = AB1_timestep(W0, P0, nu, h, k)
    P1 = gauss_seidel(W1, h, 10)
    W0, P0 = W1, P1
    time += k


psi = P1

u = np.zeros((m+2, m+2), dtype=float) # x-components
v = np.zeros((m+2, m+2), dtype=float) # y-components

for i in range(1, m+1):
    for j in range(1, m+1):
        u[j][i] = (psi[j+1][i]-psi[j-1][i])/(2*h)
        v[j][i] = -(psi[j][i+1]-psi[j][i-1])/(2*h)

for i in range(0, m+2):
    u[m+1][i] = 1.0


xdom = np.linspace(0.0, 1.0, num=m+2, endpoint=True)    # x-domain
ydom = np.linspace(0.0, 1.0, num=m+2, endpoint=True)    # y-domain
X, Y = np.meshgrid(xdom, ydom)                          # Grid of x,y values

# Plot velocity field
plt.quiver(X, Y, u, v, alpha=.5)
plt.quiver(X, Y, u, v, edgecolor='k', facecolor='None', linewidth=.5)
plt.show()

# Plot stream lines
fig, ax0 = plt.subplots()
strm = ax0.streamplot(X, Y, u, v, color=u, linewidth=1, cmap='cool')
fig.colorbar(strm.lines)
ax0.set_title('Stream Lines')
plt.show()













