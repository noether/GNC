from scipy import linalg as la
import matplotlib.pyplot as pl
import matplotlib.mlab as mlab
import numpy as np

# Take pics for video or animated gif
video = 0

# Simulation parameters
tf = 15
dt = 1e-3
time = np.linspace(0, tf, tf/dt)
it = 0;
frames = 100

# States
q = np.array([0, 0, 0])
P = np.array([[1, 0, 0], \
              [0,  1, 0], \
              [0,  0, 1]])

F = np.array([[1, dt, -0.5*dt*dt], \
              [0,  1,-dt], \
              [0,  0, 1]])
G = np.array([0.5*dt*dt, dt, 0])


# Measurements
GPS = 0.0
VEL = 0.0

acc_sigma = 0.01
acc_bias = 0.7
gps_sigma = 2
radar_sigma = 0.01

H1obs = np.array([1, 0, 0])
H2obs = np.array([0, 1, 0])
H3obs = np.array([[1, 0, 0], \
               [0, 1, 0]])

H1mes = np.array([1])
H2mes = np.array([1])
H3mes = np.array([[1, 0], \
                  [0, 1]])

Q = np.outer(G*acc_sigma,acc_sigma*G.transpose())
P1ym = (H1mes*gps_sigma).dot(gps_sigma*H1mes.transpose())
P2ym = (H2mes*radar_sigma).dot(radar_sigma*H2mes.transpose())
P3ym = (H3mes.dot(np.array([gps_sigma, radar_sigma]))).dot((np.array([gps_sigma, radar_sigma]).transpose()).dot(H3mes.transpose()))


# Data log
acc_log = np.zeros(time.size)
q_log = np.zeros((time.size, q.size))
P_log = np.zeros((time.size, P.size))

# Plotting stuff
pl.close("all")
pl.ion()
fig, axis = pl.subplots(4, 1)
fig.tight_layout()
if video == 1:
    mng = pl.get_current_fig_manager()
    mng.window.showMaximized()
    pl.pause(2)

xpl, xvl, xbl = 15, 5, 1.5
ypl, yvl, ybl = 0.5, 1.5, 8

xlimits0 = np.linspace(-xpl, xpl, 300)
xlimits1 = np.linspace(-xvl, xvl, 300)
xlimits2 = np.linspace(-xbl, xbl, 300)

for t in time:

    acc = np.random.normal(acc_bias, acc_sigma)
    
    # Propagation
    q = F.dot(q) + G.dot(acc)
    P = F.dot(P).dot(F.transpose()) + Q

    # Correction (measurements)
    if it%1000 == 0:
        q_saved = q
        P_saved = P

        S = H1obs.dot(P).dot(H1obs.transpose()) + P1ym
        if np.size(S) == 1:
            K = P.dot(H1obs.transpose())/S
            P = P - np.outer(K, H1obs.dot(P))
        else:
            K = P.dot(H1obs.transpose()).dot(la.inv(S))
            P = P - K.dot(H1obs.dot(P))
        
        q = q + K*(H1mes.dot(GPS) - H1obs.dot(q))
        if not np.all(la.eigvals(P) > 0):
            q = q_saved
            P = P_saved

    # Animation
    if it%frames == 0:
        axis[0].clear()
        axis[0].grid("on")
        pgauss = mlab.normpdf(xlimits0, q[0], np.sqrt(P[0,0]))
        axis[0].plot(xlimits0, pgauss)
        axis[0].fill_between(xlimits0, pgauss, color='cyan')
        axis[0].set_xlim([-xpl, xpl])
        axis[0].set_ylim([0, ypl])
        axis[0].set_yticks([0, 0.5*ypl, ypl])
        axis[0].set_title("Estimated position")
        axis[0].set_xlabel("[m]")
        axis[0].arrow(0, 0, 0, ypl, \
                head_width=0.05, head_length=0.1, fc='k', ec='k')

        axis[1].clear()
        axis[1].grid("on")
        vgauss = mlab.normpdf(xlimits1, q[1], np.sqrt(P[1,1]))
        axis[1].plot(xlimits1, vgauss)
        axis[1].fill_between(xlimits1, vgauss, color='cyan')
        axis[1].set_xlim([-xvl, xvl])
        axis[1].set_ylim([0, yvl])
        axis[1].set_yticks([0, 0.5*yvl, yvl])
        axis[1].set_title("Estimated velocity")
        axis[1].set_xlabel("[m/s]")
        axis[1].arrow(0, 0, 0, yvl, \
                head_width=0.05, head_length=0.1, fc='k', ec='k')

        axis[2].clear()
        axis[2].grid("on")
        bgauss = mlab.normpdf(xlimits2, q[2], np.sqrt(P[2,2]))
        axis[2].plot(xlimits2, bgauss)
        axis[2].fill_between(xlimits2, bgauss, color='cyan')
        axis[2].set_xlim([-xbl, xbl])
        axis[2].set_ylim([0, ybl])
        axis[2].set_yticks([0, 0.5*ybl, ybl])
        axis[2].set_title("Estimated accelerometer's bias")
        axis[2].set_xlabel("[m/$s^2$]")
        axis[2].arrow(acc_bias, 0, 0, ybl, \
                head_width=0.05, head_length=0.1, fc='k', ec='k')

        axis[3].clear()
        axis[3].grid("on")
        axis[3].plot(time[0:it], acc_log[0:it], 'r')
        axis[3].plot(time[0:it], q_log[0:it,2], 'b')
        axis[3].set_xlim([0, tf])
        axis[3].set_ylim([0, 1.5*acc_bias])
        axis[3].set_title("Accelerometer readings & estimated bias")
        axis[3].set_xlabel("[m/$s^2$]")

        pl.pause(0.001)

        if video == 1:
            namepic = '%i'%it
            digits = len(str(it))
            for j in range(0, 5-digits):
                namepic = '0' + namepic
            pl.savefig("./images/%s.png"%namepic)

    # Log
    q_log[it,:] = q
    P_log[it,:] = P.reshape((1,9))
    acc_log[it] = acc

    it = it + 1

fig2, axis2 = pl.subplots()
axis2.clear()
axis2.grid("on")
pgauss = mlab.normpdf(xlimits0, 5, np.sqrt(P[0,0]))
axis2.plot(xlimits0, pgauss)
axis2.fill_between(xlimits0, pgauss, color='cyan')
axis2.set_xlim([-xpl, xpl])
axis2.set_ylim([0, 0.3])
axis2.set_yticks([0, 0.3*ypl, 0.6*ypl])
axis2.set_title("Estimated position")
axis2.set_xlabel("[m]")

axis2.annotate(s='', xy=(-np.sqrt(P[0,0])+5,0.27), xytext=(np.sqrt(P[0,0])+5,0.27), arrowprops=dict(arrowstyle='<->'))
axis2.annotate(s='', xy=(5,0), xytext=(5,np.max(pgauss)), arrowprops=dict(arrowstyle='<-'))
axis2.annotate('$\sigma_p$', xy=(5, 0.275), xytext=(5, 0.275))
axis2.annotate('$\hat p$', xy=(4.2, 0.17), xytext=(4.2, 0.17))
