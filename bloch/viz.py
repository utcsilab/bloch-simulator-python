#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
from matplotlib import animation
from IPython.display import HTML





ARROW_COLORS = list(mcolors.TABLEAU_COLORS.keys())


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_patch(arrow)
    return arrow


setattr(Axes3D, "arrow3D", _arrow3D)


def visualize_magnetization_anim(
    RF,
    B0,
    mx,
    my,
    mz,
    accel=1,
    plot_sum=False,
    fig=None,
    in_notebook=False,
    interval_ms=20,
    notebook_mode="jshtml",   # "jshtml" or "html5"
):
    """
    Returns:
      - if in_notebook=False: (fig, ax, ani)
      - if in_notebook=True:  (fig, ax, ani, html) where html is display-ready (HTML object)

    Notes:
      - In notebooks, you must keep a reference to `ani` (e.g., store in a list),
        otherwise it may be garbage-collected and stop rendering.
      - For 3D artists, blit=False is correct.
    """

    if len(mx.shape) == 1:
        mx = mx[None, :]
        my = my[None, :]
        mz = mz[None, :]

    if plot_sum:
        mx = np.vstack((mx, np.sum(mx, 0)))
        my = np.vstack((my, np.sum(my, 0)))
        mz = np.vstack((mz, np.sum(mz, 0)))

    T = mx.shape[1]

    if RF is None:
        RF = np.zeros((T,), dtype=complex)
    if B0 is None:
        B0 = np.zeros((T,))

    if fig is None:
        fig = plt.figure(figsize=(8, 8))
    fig.clf()
    ax = fig.add_subplot(111, projection="3d")

    sc = 1 / 0.85
    x0 = -sc * np.eye(3)
    x1 = sc * np.eye(3)

    for i in range(3):
        ax.plot(
            [x0[i, 0], x1[i, 0]],
            [x0[i, 1], x1[i, 1]],
            [x0[i, 2], x1[i, 2]],
            "gray",
        )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.axis("off")
    ax.view_init(elev=30.0, azim=60)

    if isinstance(accel, int):
        accel = [[0, accel]]
    accel = accel + [[T, accel[-1][1]]]

    frames = []
    for j in range(len(accel) - 1):
        start = int(accel[j][0])
        end = int(accel[j + 1][0])
        step = int(accel[j][1])
        frames.extend(range(start, end, step))

    rf_arrow = ax.arrow3D(0, 0, 0, 0, 0, 0, mutation_scale=30, ec="k", fc=ARROW_COLORS[0])
    b0_arrow = ax.arrow3D(0, 0, 0, 0, 0, 0, mutation_scale=30, ec="k", fc=ARROW_COLORS[1])

    m_arrows = []
    trail_lines = []
    for k in range(mx.shape[0]):
        col = ARROW_COLORS[(k + 2) % len(ARROW_COLORS)]
        a = ax.arrow3D(
            0, 0, 0, 0, 0, 0,
            mutation_scale=20,
            ec=col, fc=col, alpha=0.6
        )
        m_arrows.append(a)

        (ln,) = ax.plot([mx[k, 0]], [my[k, 0]], [mz[k, 0]],
                        linestyle="--", color=col)
        trail_lines.append(ln)

    def init():
        rf_arrow._xyz = (0, 0, 0); rf_arrow._dxdydz = (0, 0, 0)
        b0_arrow._xyz = (0, 0, 0); b0_arrow._dxdydz = (0, 0, 0)

        for k in range(mx.shape[0]):
            m_arrows[k]._xyz = (0, 0, 0)
            m_arrows[k]._dxdydz = (0, 0, 0)
            trail_lines[k].set_data([mx[k, 0]], [my[k, 0]])
            trail_lines[k].set_3d_properties([mz[k, 0]])

        return [rf_arrow, b0_arrow] + m_arrows + trail_lines

    def update(f):
        # RF arrow
        rf_arrow._xyz = (0, 0, 0)
        rf_arrow._dxdydz = (float(np.real(RF[f])), float(-np.imag(RF[f])), 0.0)

        # B0 arrow
        b0_arrow._xyz = (0, 0, 0)
        b0_arrow._dxdydz = (0.0, 0.0, float(B0[f]))

        # magnetization arrows & trails
        for k in range(mx.shape[0]):
            m_arrows[k]._xyz = (0, 0, 0)
            m_arrows[k]._dxdydz = (float(mx[k, f]), float(my[k, f]), float(mz[k, f]))

            trail_lines[k].set_data(mx[k, : f + 1], my[k, : f + 1])
            trail_lines[k].set_3d_properties(mz[k, : f + 1])

        # IMPORTANT: no fig.canvas.draw/flush here!
        return [rf_arrow, b0_arrow] + m_arrows + trail_lines

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=interval_ms,
        blit=False,      # correct for 3D
        repeat=False
    )

    if in_notebook:
        # Most reliable across notebook backends:
        if notebook_mode == "html5":
            html = HTML(ani.to_html5_video())
        else:
            html = HTML(ani.to_jshtml())
        return fig, ax, ani, html

    return fig, ax, ani


if __name__ == "__main__":
    from . import bloch
    from .min_time_gradient import minimum_time_gradient


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

    B1 = np.zeros((int(np.round(sim_time/dt)),), dtype=complex)
    G = np.zeros((int(np.round(sim_time/dt)),))
    B1[:N] = b1
    G[N:N+len(g)] = g
    B1[N+len(g):N+len(g)+N2] = b2
    G[N+len(g)+N2:N+len(g)+N2+len(g)] = g
    mx, my, mz = bloch(B1, G, dt, 100e-4, 100e-5, 0, np.linspace(-.5, .5, 15)[None,:], 2, mx=1, my=0, mz=0)



    fig, ax, ani = visualize_magnetization_anim(B1, G*x, mx, my, mz, accel=1, plot_sum=False, in_notebook=False, interval_ms=20)

    #from matplotlib.animation import FFMpegWriter
    #ani.save("bloch_x_0p05.gif", writer="pillow", fps=30, dpi=80)
    #writer = FFMpegWriter(fps=30, bitrate=1200)
    #ani.save("out.mp4", writer=writer)
    #plt.close(fig)


    plt.show()

