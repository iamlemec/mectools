# endy - high dimensional array tools

import numpy as np

# digitize nd array
def digitize(x, bins):
    N = len(bins)
    d = np.zeros_like(x, dtype=np.int)
    for i in range(N):
        d[x>=bins[i]] = i
    return d

# choose along a given axis
def choose(d, f, axis=0):
    return np.choose(d.swapaxes(0, axis), f.swapaxes(0, axis)).swapaxes(0, axis)

# interpolate non-flat domains on a given axis
def interp(x1, x0, f0, axis=0):
    N = len(x0)
    d1 = digitize(x1, x0)
    d2 = np.minimum(N-1, d1+1)
    d1[d1==d2] -= 1 # only happens on top bin
    q = (x1-x0[d1])/(x0[d2]-x0[d1])
    g1 = choose(d1, f0, axis=axis)
    g2 = choose(d2, f0, axis=axis)
    f1 = (1-q)*g1 + q*g2
    return f1

# insert added possibly flat dimensions
def expand(x, dims=1, axis=-1):
    if type(dims) is int:
        dims = (1,)*dims
    if axis < 0:
        axis += len(x.shape) + 1
    shape = x.shape[:axis] + dims + x.shape[axis:]
    return x.reshape(shape)

# project non-flat ranges on bins
def project(bins, xp=None, xlo=None, xhi=None, axis=-1):
    # if point given, assume additively scaled ranges around each bin
    if xp is not None:
        nd = len(xp.shape)
        db = np.diff(bins)
        dbx = expand(expand(db, nd-axis-1, -1), axis, 0)
        xlo, xhi = xp - dbx/2, xp + dbx/2

    # general code for xlo, xhi
    blo, bhi = expand(bins[:-1], xlo.ndim, 0), expand(bins[1:], xhi.ndim, 0)
    xlo, xhi = expand(xlo, 1, -1), expand(xhi, 1, -1)
    upper = np.maximum(0, bhi - np.maximum(xlo, blo))
    lower = np.maximum(0, bhi - np.maximum(xhi, blo))
    total = upper - lower
    total[...,[ 0]] += np.maximum(0, blo[..., 0] - xlo) - np.maximum(0, blo[..., 0] - xhi)
    total[...,[-1]] += np.maximum(0, xhi - bhi[...,-1]) - np.maximum(0, xlo - bhi[...,-1])
    return total/(bhi-blo)
