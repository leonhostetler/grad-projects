#! /usr/bin/env python
"""
This script solves the advection equation
    u_t + a*u_x = 0
with periodic boundary conditions and initial condition
    u(x,0) = f(x)
using the Upwind finite difference method. This method requires a > 0.

CMSE 821: Numerical Methods for Differential Equations
Michigan State University
Leon Hostetler, Apr. 23, 2018

"""
from __future__ import division, print_function
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

###############################################################
#                    FUNCTION DEFINITIONS
###############################################################

def initial(x):
    """
    Set the initial state of the system here
    """
    return np.exp(-20*(x-2)**2) + np.exp(-(x-5)**2)


def Upwind(U, a, k, h):
    """
    For the advection equation, this function computes the solution at
    the next time step using the Lax-Wendroff method.

    Takes in:
        U: The solution at the previous time step
        a: The wave speed
        k: The temporal spacing
        h: The spatial spacing
        
    Returns:
        V: The solution at the next time step

    """
    V = np.zeros(len(U))
    c = a*k/h
    m = int(1/h - 1)

    # Periodic boundary conditions
    V[0] = U[0] - c*(U[1]-U[m])/2.0 + c*(U[1]-2*U[0]+U[m])/2.0
    V[-1] = V[0]

    for j in range(1, len(U)-1):
        V[j] = U[j] - c*(U[j+1]-U[j-1])/2.0 + c*(U[j+1]-2*U[j]+U[j-1])/2.0

    return V


###############################################################
#                        MAIN PART
###############################################################

# FDM parameters
x0 = 0.0                                    # The left endpoint of your spatial domain
xf = 25.0                                   # The right endpoint of your spatial domain
t0 = 0.0                                    # The starting time
tf = 17.0                                   # The ending time
a = 1.0                                     # wave speed
h = 0.05                                    # spatial spacing
k = 0.8*h/a                                 # temporal spacing

# Input verification
if np.abs(a*k/h) > 1.0:
    raise Exception('Unstable! Choose different FDM parameters!')

if a < 0:
    print("WARNING: Upwind method is unstable for a < 0!")

# Discretization
m = int((xf-x0)/h - 1)                                          # The number of interior lattice points
x = np.linspace(x0, xf, num=m+2, endpoint=True, dtype=float)    # Spatial domain, including boundary points

# Time steps
U0 = initial(x)           # The initial state
sol_list = [U0]           # The list of solutions--one for each time step

time = t0
while time < tf:          # Loop through all time steps
    U = Upwind(sol_list[-1], a, k, h)
    sol_list.append(U)
    time += k


###############################################################
#                        Plotting
###############################################################
"""
Set plot_simple to True to show a plot of the initial and final states
"""
plot_simple = True
if plot_simple:
    plt.rc('text', usetex=True)
    plt.plot(x, sol_list[0], label="Initial State")
    plt.plot(x, sol_list[-1], label="Final State")
    plt.legend(loc=2)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$U(x)$")
    plt.title("Upwind Method")
    plt.show()


"""
Set plot_animate to True to show an animation of the solution
as it evolves in time.
"""
plot_animate = True
if plot_animate:

    fig, ax = plt.subplots()
    line, = ax.plot(x, U0)
    plt.rc('text', usetex=True)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$U(x)$")
    plt.title("Upwind Method")

    def init():                         # Initialize for blitting
        line.set_ydata(sol_list[0])
        return line,

    def animate(i):
        line.set_ydata(sol_list[i])     # Update the data
        return line,

    ani = FuncAnimation(fig, animate, frames=len(sol_list), interval=20,
                        init_func=init, blit=True, repeat=False)

    plt.show()

