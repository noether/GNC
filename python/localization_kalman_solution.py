from scipy import linalg as la
import matplotlib.pyplot as pl
import matplotlib.mlab as mlab
import numpy as np

# Simulation parameters
tf = 50
dt = 1e-3
time = np.linspace(0, tf, tf/dt)
it = 0;
frames = 100

# States
X = np.array([-5, -5])
P = np.array([[1, 0], \
              [0, 1]])

F = np.array([[1, 0], \
              [0, 1]])

G = np.array([[dt, 0], \
              [0, dt]])

# Measurements
radar_sigma = 1
vel_sigma = np.array([10, 10])

Q = (G*vel_sigma).dot(vel_sigma*G.transpose())

# Data log
X_log = np.zeros((time.size, X.size))
P_log = np.zeros((time.size, P.size))

# Plotting stuff
pl.close("all")
fig, axis = pl.subplots(3, 1)

xpx, xpy = 10, 20
ypx, ypy = 10, 20

xlimits0 = np.linspace(-xpx, xpx, 300)
xlimits1 = np.linspace(-ypx, ypx, 300)

# Initial position
p = np.array([0, 0])

for t in time:

    v = np.array([5*np.sin(t), 10*np.cos(t)])
    p = F.dot(p) + G.dot(v)

    # Propagation
    X = F.dot(X) + G.dot(v)
    P = F.dot(P).dot(F.transpose()) + Q

    # Correction (measurements)
    if it%1000 == 0:

        e = la.norm(p) - la.norm(X)
        H = X/la.norm(X)
        R = (H*radar_sigma).dot(radar_sigma*H.transpose())
        S = H.dot(P).dot(H.transpose()) + R
        K = P.dot(H.transpose())/S

        X = X + K*e
        P = P - np.outer(K, H.dot(P))

    # Animation
    if it%frames == 0:
        axis[0].clear()
        axis[0].grid("on")
        pgauss = mlab.normpdf(xlimits0, X[0], np.sqrt(P[0,0]))
        axis[0].plot(xlimits0, pgauss)
        axis[0].fill_between(xlimits0, pgauss, color='cyan')
        axis[0].set_xlim([-xpx, xpx])
        axis[0].set_ylim([0, ypx])
        axis[0].set_yticks([0, 0.5*ypx, ypx])
        axis[0].set_title("Estimated position x")
        axis[0].set_xlabel("[m]")
        axis[0].arrow(p[0], 0, 0, ypx, \
                head_width=0.05, head_length=0.1, fc='k', ec='k')

        axis[1].clear()
        axis[1].grid("on")
        vgauss = mlab.normpdf(xlimits1, X[1], np.sqrt(P[1,1]))
        axis[1].plot(xlimits1, vgauss)
        axis[1].fill_between(xlimits1, vgauss, color='cyan')
        axis[1].set_xlim([-xpy, xpy])
        axis[1].set_ylim([0, ypy])
        axis[1].set_yticks([0, 0.5*ypy, ypy])
        axis[1].set_title("Estimated position y")
        axis[1].set_xlabel("[m]")
        axis[1].arrow(p[1], 0, 0, ypy, \
                head_width=0.05, head_length=0.1, fc='k', ec='k')

        axis[2].clear()
        axis[2].grid("on")
        axis[2].set_xlim([-xpx, xpx])
        axis[2].set_ylim([-xpy, xpy])
        axis[2].arrow(0,0,X[0],X[1],head_width=0.5, head_length=1, facecolor='black',length_includes_head=True)
        axis[2].plot(p[0],p[1], 'ok',markersize=6)

        pl.pause(0.001)

    # Log
    X_log[it,:] = X
    P_log[it,:] = P.reshape((1,4))

    it = it + 1
