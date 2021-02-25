#!/usr/bin/env python

# bloch visualization 
# based on Matlab code from Miki Lustig
# Jon Tamir, 2020

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors

ARROW_COLORS = list(mcolors.TABLEAU_COLORS.keys())


# based on https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._xyz = (x,y,z)
        self._dxdydz = (dx,dy,dz)

    def draw(self, renderer):
        x1,y1,z1 = self._xyz
        dx,dy,dz = self._dxdydz
        x2,y2,z2 = (x1+dx,y1+dy,z1+dz)

        xs, ys, zs = proj_transform((x1,x2),(y1,y2),(z1,z2), renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(renderer)
        
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add a 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D,'arrow3D',_arrow3D) # this line should not be indented


def visualize_magnetization(RF, B0, mx, my, mz, accel=1, plot_sum=False, fig=None):
    '''Visualizes the magnetization of a given B1, B0, and magnetization vector
    
    Args:
        RF (np.array): B1 field (real is x axis, imag is y axis)
        B0 (np.array): B0 field
        mx (np.array): MxN Transverse Mx component for M spins at N time points
        my (np.array): MxN Transverse My component for M spins at N time points
        mz (np.array): MxN Longitudinal Mz component for M spins at N time points
        accel (int): plot acceleration. Default: 1  
        plot_sum (bool): True to also plot the net magnetization across the M spins
        fig (plt.figure): existing figure object
    '''

    if len(mx.shape) == 1:
        mx = mx[None,:]
        my = my[None,:]
        mz = mz[None,:]
        
    if plot_sum:
        mx = np.vstack((mx, np.sum(mx, 0)))
        my = np.vstack((my, np.sum(my, 0)))
        mz = np.vstack((mz, np.sum(mz, 0)))

    if fig is None:
        fig = plt.figure(figsize=(8,8))
        
    if type(accel) == type(1):
        accel = [[0, accel]]            
    accel = accel + [[np.shape(mx)[1], accel[-1][1]]]

    if RF is None:
        RF = np.zeros((mx.shape[1],), dtype=np.complex)
    if B0 is None:
        B0 = np.zeros((mx.shape[1],))

    sc = 1/0.85
    x0 = -sc * np.eye(3)
    x1 = sc * np.eye(3)

    ax = fig.add_subplot(111, projection='3d')

    for i in range(3):
        ax.plot([x0[i, 0], x1[i, 0]], [x0[i, 1], x1[i, 1]], [x0[i,2], x1[i,2]], 'gray')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.axis('off')

    fig.show()
    fig.canvas.draw()
    ax.view_init(elev=30., azim=60)
    
    for j in range(len(accel)-1):
        acc_start = accel[j][0]
        acc_end = accel[j+1][0]
        acc = accel[j][1]
        for i in range(acc_start, acc_end, acc):  

            if i > 0:
                ax.artists.pop(0)
            ax.arrow3D(0., 0., 0.,
                   float(np.real(RF[i])), float(-np.imag(RF[i])), 0.,
                   mutation_scale=30,
                   ec ='k',
                   fc=ARROW_COLORS[0])

            if i > 0:
                ax.artists.pop(0)
            ax.arrow3D(0., 0., 0.,
                   0., 0., float(B0[i]),
                   mutation_scale=30,
                   ec ='k',
                   fc=ARROW_COLORS[1])
            
            for k in range(len(mx)):
                if i > 0:
                    ax.artists.pop(0)
                ax.arrow3D(0., 0., 0.,
                       float(mx[k,i]), float(my[k,i]), float(mz[k,i]),
                       mutation_scale=30,
                       ec=ARROW_COLORS[(k+2) %len(ARROW_COLORS)],
                       fc=ARROW_COLORS[(k+2) %len(ARROW_COLORS)],
                          alpha=.6)
                if i > 0:
                    ax.lines.pop(3)
                lines = ax.plot(mx[k,:i], my[k,:i], mz[k,:i], linestyle='--', color=ARROW_COLORS[(k+2) %len(ARROW_COLORS)])
            fig.canvas.draw()


if __name__ == "__main__":
    from bloch.bloch import bloch
    from min_time_gradient import minimum_time_gradient


    gamma_bar = 4257
    gamma = 2 * np.pi * gamma_bar
    dt = 4e-6
    B1max = 1
    Smax = 15000
    Gmax = 4

    sim_time = .0025 # s

    flip = 90
    flip_rad = flip * np.pi/180
    b1_time = flip_rad / B1max / gamma
    N = int(np.ceil(b1_time / dt))
    b1 = np.ones((N,)) * B1max

    flip = 180
    flip_rad = flip * np.pi/180
    b2_time = flip_rad / B1max / gamma
    N2 = int(np.ceil(b2_time / dt))
    b2 = np.ones((N2,)) * B1max * 1j

    x = 1
    area = .99 / x / gamma_bar
    g = minimum_time_gradient(area, Gmax, Smax, dt)

    B1 = np.zeros((int(np.round(sim_time/dt)),), dtype=np.complex)
    G = np.zeros((int(np.round(sim_time/dt)),))
    B1[:N] = b1
    G[N:N+len(g)] = g
    B1[N+len(g):N+len(g)+N2] = b2
    G[N+len(g)+N2:N+len(g)+N2+len(g)] = g
    mx, my, mz = bloch(B1, G, dt, 100e-4, 100e-5, 0, np.linspace(-.5, .5, 15)[None,:], 2, mx=1, my=0, mz=0)



    visualize_magnetization(B1, G*x, mx, my, mz, accel=5)

