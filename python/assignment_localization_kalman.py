from scipy import linalg as la
import matplotlib.pyplot as pl
import matplotlib.mlab as mlab
import numpy as np

# Simulation parameters
tf = 50 # final time in seconds
dt = 1e-3 # step time in seconds
time = np.linspace(0, tf, tf/dt) # vector with all the time events 0, 0.001, 0.002....
it = 0 # current iteration in the for loop
frames = 100 # We draw the plots once per 100 iterations, i.e., every 0.1 seconds.

# Set initial conditions for the Kalman Filter
q_hat = np.random.uniform(-10, 10, (2,1)) # Random 2x1 vector from -10 to 10
P = np.array([[5, 0], \
              [0, 5]])

# Standard deviations for the measurements (we consider them constant for the whole simulation)
sigma_r =  # range measurement
sigma_u =  # velocities as inputs

# Data log
q_hat_log = np.zeros((time.size, X_hat.size))
P_log = np.zeros((time.size, P.size))

# Plotting stuff
pl.close("all")
fig, axis = pl.subplots(3, 1) # Three plots for the Gaussians p_x and p_y, and for the robot's position

xpx, xpy = 10, 20 # Some limits for plotting the Gaussians
ypx, ypy = 10, 20
xlimits0 = np.linspace(-xpx, xpx, 300)
xlimits1 = np.linspace(-ypx, ypx, 300)

# Initial position
p = np.array([[0], [0]])

for t in time:

    # Simulation of the mobile robot
    v = np.array([[5*np.sin(t)], [10*np.cos(t)]])
    p = p + v*dt

    # Simulation with the Lyapunov controller. Do not forget to comment the above two lines
    # c = # Controller
    # p = p + c*dt

    # Discrete linear KALMAN filter
    if it%1 == 0:
        # Transition matrices
        F = # set F

        G = # set G

        # Process noise
        Q = # set Q

        # Dynamics of the Gaussian states (mean and variances)
        u = # set u

        q_hat = # Eq. (11)
        P = # Eq. (17)

        # Update after a range measurement (any other suggestions about when to update?)
        if it%1000 == 0:

            H = # Eq. (19) or Eq. (29)
            Pym = # set P_ym.
            K = # Eq. (24)

            q_hat_u = # Eq. (27)
            P_u = # Eq. (26)

            q_hat = q_hat_u
            P = P_u

    # Animation
    if it%frames == 0:

        # We plot the Gaussian distribution for p_x
        axis[0].clear()
        axis[0].grid("on")
        pgauss = mlab.normpdf(xlimits0, X_hat[0], np.sqrt(P[0,0]))
        axis[0].plot(xlimits0, pgauss)
        axis[0].fill_between(xlimits0, pgauss, color='cyan')
        axis[0].set_xlim([-xpx, xpx])
        axis[0].set_ylim([0, ypx])
        axis[0].set_yticks([0, 0.5*ypx, ypx])
        axis[0].set_title("Estimated position x")
        axis[0].set_xlabel("[m]")
        axis[0].arrow(p[0], 0, 0, ypx, \
                head_width=0.05, head_length=0.1, fc='k', ec='k')

        # We plot the Gaussian distribution for p_y
        axis[1].clear()
        axis[1].grid("on")
        vgauss = mlab.normpdf(xlimits1, X_hat[1], np.sqrt(P[1,1]))
        axis[1].plot(xlimits1, vgauss)
        axis[1].fill_between(xlimits1, vgauss, color='cyan')
        axis[1].set_xlim([-xpy, xpy])
        axis[1].set_ylim([0, ypy])
        axis[1].set_yticks([0, 0.5*ypy, ypy])
        axis[1].set_title("Estimated position y")
        axis[1].set_xlabel("[m]")
        axis[1].arrow(p[1], 0, 0, ypy, \
                head_width=0.05, head_length=0.1, fc='k', ec='k')

        # We plot the estimation (with an arrow) and the actual position (with a big dot) of the robot
        axis[2].clear()
        axis[2].grid("on")
        axis[2].set_xlim([-xpx, xpx])
        axis[2].set_ylim([-xpy, xpy])
        axis[2].arrow(0,0,X_hat[0],X_hat[1],head_width=0.5, head_length=1, facecolor='black',length_includes_head=True)
        axis[2].plot(p[0],p[1], 'ok',markersize=6)

        # Otherwise, it might not draw our plots
        pl.pause(0.001)

    # Log
    q_hat_log[it,:] = q_hat.reshape((1,2))
    P_log[it,:] = P.reshape((1,4))

    it = it + 1
