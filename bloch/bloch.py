import numpy as np
import scipy as sp

from bloch.bloch_simulator import bloch_c

from bloch.bloch_processing import NUMBER
from bloch.bloch_processing import process_gradient_argument, process_time_points, process_off_resonance_arguments
from bloch.bloch_processing import process_positions, process_magnetization, reshape_matrices
from bloch.bloch_processing import process_relaxations

def bloch(b1, gr, tp, t1, t2, df, dp, mode, mx=None, my=None, mz=None):
    """
    Bloch simulation of rotations due to B1, gradient and
    off-resonance, including relaxation effects.  At each time
    point, the rotation matrix and decay matrix are calculated.
    Simulation can simulate the steady-state if the sequence
    is applied repeatedly, or the magnetization starting at m0.

    INPUT:
            b1 = (1xM) RF pulse in G.  Can be complex.
            gr = ((1,2,or 3)xM) 1,2 or 3-dimensional gradient in G/cm.
            tp = (1xM) time duration of each b1 and gr point, in seconds,
                            or 1x1 time step if constant for all points
                            or monotonically INCREASING endtime of each
                            interval..
            t1 = T1 relaxation time in seconds.
            t2 = T2 relaxation time in seconds.
            df = (1xN) Array of off-resonance frequencies (Hz)
            dp = ((1,2,or 3)xP) Array of spatial positions (cm).
                    Width should match width of gr.
            mode= Bitmask mode:
                    Bit 0:  0-Simulate from start or M0, 1-Steady State
                    Bit 1:  1-Record m at time points.  0-just end time.

    (optional)
            mx,my,mz (NxP) arrays of starting magnetization, where N
                    is the number of frequencies and P is the number
                    of spatial positions.

    OUTPUT:
            mx,my,mz = NxP arrays of the resulting magnetization
                            components at each position and frequency.
    """
    if isinstance(b1, NUMBER):
        b1 = np.ones(1) * b1
    if b1.ndim == 1:
        b1 = np.expand_dims(b1, 0)
    ntime = b1.size[1]

    assert gr.ndim == 2 and gr.shape[0] <= 3 and gr.shape[1] == ntime, 'gr is of the wrong shape.'

    tp_flag = False
    tp_flag |= isinstance(tp, NUMBER)
    tp_flag |= (tp.ndim == 1)
    if tp.ndim == 2:
        tp_flag |= (tp.shape[0] == 1) and ((tp.shape[1] == 1) or (tp.shape[1] == ntime))
    else:
        tp_flag |= False
    assert tp_flag, 'tp is of the wrong shape.'

    grx, gry, grz = process_gradient_argument(gr, ntime)
    tp = process_time_points(tp, ntime)
    df, nf = process_off_resonance_arguments(df)
    dx, dy, dz, n_pos = process_positions(dp)
    t1, t2 = process_relaxations(t1, t2, n_pos)
    mx, my, mz = process_magnetization(mx, my, mz, ntime, nf*n_pos, mode)

    if (2 & mode):
        ntout = ntime
    else:
        ntout = 1

    bloch_c(b1.real, b1.imag, grx, gry, grz, tp, ntime, t1, t2, df, nf, dx, dy, dz, n_pos, mode, mx, my, mz)

    reshape_matrices(mx, my, mz, ntout, n_pos, nf)
    return mx, my, mz
