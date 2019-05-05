#! /usr/bin/env python
"""
This script solves the advection equation
    u_t + a*u_x = 0
with periodic boundary conditions and initial condition
    u(x,0) = f(x)
using the Lax-Wendroff, Lax-Friedrichs, and Upwind finite difference methods

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
    return np.sin(2.0*np.pi*x)


def exact(x, t, a):
    """
    Set the exact solution here for error analysis
    """
    return np.sin(2.0*np.pi*(x-a*t))


def Lax_Friedrichs(U, a, k, h):
    """
    For the advection equation, this function computes the solution at
    the next time step using the Lax-Friedrichs method.

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
    V[0] = U[0] - c*(U[1]-U[m])/2.0 + (U[1]-2*U[0]+U[m])/2.0
    V[-1] = V[0]

    for j in range(1, len(U)-1):
        V[j] = U[j] - c*(U[j+1]-U[j-1])/2.0 + (U[j+1]-2*U[j]+U[j-1])/2.0

    return V


def Lax_Wendroff(U, a, k, h):
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
    V[0] = U[0] - c*(U[1]-U[m])/2.0 + c*c*(U[1]-2*U[0]+U[m])/2.0
    V[-1] = V[0]

    for j in range(1, len(U)-1):
        V[j] = U[j] - c*(U[j+1]-U[j-1])/2.0 + c*c*(U[j+1]-2*U[j]+U[j-1])/2.0

    return V


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


def norm_l2(vec, eps):
    """
    Takes a vector vec and the lattice spacing eps and returns
    the l2-norm of the given vector
    """
    return np.sqrt(eps*np.sum(np.square(vec)))



def verify(h, k, a):
    """
    Stability verification for given h, k, and a
    """
    if np.abs(a*k/h) > 1.0:
        raise Exception('Unstable! Choose different FDM parameters!')

    if a < 0:
        print("WARNING: Upwind method is unstable for a < 0!")



def fdm_solve(x0,xf,h,t0,tf,k,a,method):
    """
    Solves the advection equation using one of the finite difference methods

    Takes in:
        x0: The left boundary point
        xf: The right boundary point
        h: The spatial spacing
        t0: The initial time
        tf: The ending time
        k: The temporal spacing
        a: The wave speed
        method: The FDM to use

    Returns:
        x: The spatial domain vector (including boundary points) for this value of h
        sol_list: The list of solutions for this value of h--one for each time step
        kerror: Temporal error
        herror: Spatial error
    """
    m = int((xf-x0)/h - 1)                                          # The number of interior lattice points
    x = np.linspace(x0, xf, num=m+2, endpoint=True, dtype=float)    # Spatial domain
    U0 = initial(x)                                                 # The initial state
    sol_list = [U0]
    t_err_fdm, t_err_exa = [], []
    t_err = []

    time = t0
    while time < tf:                                                # Loop through all time steps
        if method==1:
            U = Lax_Friedrichs(sol_list[-1], a, k, h)
        elif method==2:
            U = Lax_Wendroff(sol_list[-1], a, k, h)
        elif method==3:
            U = Upwind(sol_list[-1], a, k, h)
        else:
            raise Exception('Invalid method selected!')

        sol_list.append(U)
        time += k

        # Track the error through time for a fixed spatial point e.g. x = xf
        t_err.append(U[-1] - exact(xf, time, a))

    # Temporal error
    kerror = norm_l2(np.asarray(t_err), k)

    # Spatial error--the error of the solution at a fixed point in time
    mid = int(len(sol_list[-1])/2)
    herror = norm_l2(sol_list[mid] - exact(x, t0+mid*k, a), h)

    return x, sol_list, kerror, herror


###############################################################
#                        MAIN PART
###############################################################

# FDM parameters
x0 = 0.0                                    # The left endpoint of your spatial domain
xf = 2.0                                    # The right endpoint of your spatial domain
t0 = 0.0                                    # The starting time
tf = 2.0                                    # The ending time
a = 1.0                                     # wave speed

method = 3                                  # 1 = Lax-Friedrichs, 2 = Lax-Wendroff, 3 = Upwind

print("Advection equation parameters:")
print("------------------------------")
print(" Spatial domain: [",x0,", ", xf,"]")
print("Temporal domain: [",t0,", ", tf,"]")
print("     Wave speed: ", a)

if method==1:
    title = "Lax-Friedrichs Method"
    print("\nUsing ", title)
elif method==2:
    title = "Lax-Wendroff Method"
    print("\nUsing ", title)
elif method==3:
    title = "Upwind Method"
    print("\nUsing ", title)


# Calculate discretization errors, or not
error_test = True
if error_test:
    print("\nCalculating discretization error...")

    h_list = [0.01, 0.005, 0.001,]              # spatial spacing
    k_list = [0.99*h/a for h in h_list]         # temporal spacing
    h_errors, k_errors = [], []

    for index, h in enumerate(h_list):
        k = k_list[index]
        verify(h, k, a)                         # Input verification
        x, sol_list, k_error, h_error = fdm_solve(x0,xf,h,t0,tf,k,a,method)
        h_errors += [h_error]
        k_errors += [k_error]

    # Get the slopes for the spatial discretization errors
    logE2, logE1 = np.log10(np.abs(h_errors[-1])), np.log10(np.abs(h_errors[0]))
    logh2, logh1 = np.log10(np.abs(h_list[-1])), np.log10(np.abs(h_list[0]))
    p = (logE2-logE1)/(logh2-logh1)
    print("\tSpatial error rate: ", p)

    # Get the slopes for the temporal discretization errors
    logE2, logE1 = np.log10(np.abs(k_errors[-1])), np.log10(np.abs(k_errors[0]))
    logh2, logh1 = np.log10(np.abs(k_list[-1])), np.log10(np.abs(k_list[0]))
    p = (logE2-logE1)/(logh2-logh1)
    print("\tTemporal error rate:", p)

    # Plot discretization errors versus spacing
    print("\nPlotting spatial and temporal discretization errors")
    fig, axs = plt.subplots(nrows=1, ncols=2)
    ax = axs[0]
    ax.loglog(h_list, h_errors, label=r"$||E||_2$", marker='4')
    ax.set_title('Spatial Error')
    ax.set_xlabel(r"$h$")
    ax = axs[1]
    ax.loglog(k_list, k_errors, label=r"$||E||_2$", marker='4')
    ax.set_title('Temporal Error')
    ax.set_xlabel(r"$k$")
    fig.suptitle(title)
    plt.show()


# Calculate example solution
h = 0.01                                        # spatial spacing
k = h                                           # temporal spacing
verify(h, k, a)                                 # stability verification

# Solve it
x, sol, ke, he = fdm_solve(x0,xf,h,t0,tf,k,a,method)

print("\nCalculating result for h = ",h,", k = ", k)

# Set plot_simple to True to show a plot of the initial and final states
plot_simple = True
print("Plotting initial and final state")
if plot_simple:
    plt.rc('text', usetex=True)
    plt.plot(x, sol[0], label="Initial State", linestyle='--')
    plt.plot(x, exact(x,tf,a), label="Exact Final State",
            linestyle=':', color='black', linewidth=1)
    plt.plot(x, sol[-1], label="Final State", marker='+', linestyle='none')
    plt.legend(loc=2)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$U(x,t)$")
    plt.title(title)
    plt.show()


# Set plot_animate to True to show an animation of the solution as it evolves in time.
plot_animate = True
print("Plotting time-dependent animation")
if plot_animate:
    fig, ax = plt.subplots()
    line, = ax.plot(x, sol[0])
    plt.rc('text', usetex=True)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$U(x)$")
    plt.title(title)

    def init():                    # Initialize for blitting
        line.set_ydata(sol[0])
        return line,

    def animate(i):
        line.set_ydata(sol[i])     # Update the data
        return line,

    ani = FuncAnimation(fig, animate, frames=len(sol), interval=20,
                        init_func=init, blit=True, repeat=False)

    plt.show()

print("\nAll done\n")
