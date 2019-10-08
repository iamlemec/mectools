# endy - high dimensional array tools

import numpy as np
from scipy.constants import golden

# digitize nd array
def digitize(x, bins):
    N = len(bins)
    d = np.zeros_like(x, dtype=np.int)
    for i in range(N):
        d[x>=bins[i]] = i
    return d

# choose with flat indexing logic (1 copy?)
def choose0(d, f):
    sd0 = d.shape[0]
    sd1 = np.prod(f.shape[1:])
    ic = sd1*d.reshape((sd0, -1)) + np.arange(sd1)[None, :]
    return f.flat[ic].reshape(d.shape)

# choose along a given axis
def choose(d, f, axis=0):
    # swap to 0th axis
    if axis != 0:
        d = d.swapaxes(0, axis)
        f = f.swapaxes(0, axis)

    # broadcast trailing axes
    if f.shape[1:] != d.shape[1:]:
        f = np.broadcast_to(f, (f.shape[0],)+d.shape[1:])

    # apply with 0th version
    g = choose0(d, f)

    # unswap 0th axis
    if axis != 0:
        g = g.swapaxes(0, axis)

    return g

# like choose but d omits axis of choice
def address0(d, f):
    sd0 = f.shape[0]
    sd1 = np.prod(f.shape[1:])
    ic = sd1*d.reshape((-1,)) + np.arange(sd1)
    return f.flat[ic].reshape(d.shape)

# address along a given axis
def address(d, f, axis=0):
    # swap to 0th axis
    if axis != 0:
        f = f.swapaxes(0, axis)

    # broadcast trailing axes
    if f.shape[1:] != d.shape:
        f = np.broadcast_to(f, (f.shape[0],)+d.shape)

    # apply with 0th version
    g = address0(d, f)

    return g

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

def sliceaxis(x, axis, i):
    ndim = x.ndim
    imax = x.shape[axis]
    if i < 0: i += imax
    acc = axis*(slice(None),) + (slice(i, i+1),) + (ndim-axis-1)*(slice(None),)
    return x[acc]

# project non-flat ranges on bins
# bins: grid spec (N+1)
# xlo, xhi: bin lows and his (M)
# returns: (NxM)
def project(bins, xp=None, axis=None, xlo=None, xhi=None):
    # if point given, assume additively scaled ranges around each bin
    if xp is not None:
        ndim = len(xp.shape)
        dbin = np.diff(xp, axis=axis)
        dbinx = np.concat([dbin, sliceaxis(dbin, axis, -1)], axis=axis) # repeat last axis row
        xlo, xhi = xp - dbinx/2, xp + dbinx/2

    # general code for xlo, xhi
    blo, bhi = expand(bins[:-1], xlo.ndim, 0), expand(bins[1:], xhi.ndim, 0)
    xlo, xhi = expand(xlo, 1, -1), expand(xhi, 1, -1)
    upper = np.maximum(0, bhi - np.maximum(xlo, blo))
    lower = np.maximum(0, bhi - np.maximum(xhi, blo))
    total = upper - lower
    total[...,[ 0]] += np.maximum(0, blo[..., 0] - xlo) - np.maximum(0, blo[..., 0] - xhi)
    total[...,[-1]] += np.maximum(0, xhi - bhi[...,-1]) - np.maximum(0, xlo - bhi[...,-1])
    return total/(xhi-xlo)

# make sure it's a vector
def ensure_vector(x, N):
    if np.isscalar(x):
        return x*np.ones(N)
    else:
        return np.array(x)

# binsearch over a vectorizable function
# only flat for now - could accept axis in the future
def binsearch_vec(fun, xmin, xmax, max_iter=64, bs_tol=1e-12):
    f0, f1 = fun(xmin), fun(xmax) # testing sizes
    N = len(f0)
    assert(len(f1) == N)

    # expand bounds if necessary
    x0 = ensure_vector(xmin, N)
    x1 = ensure_vector(xmax, N)

    # flip if needed
    swap = (f0 > f1)
    x0[swap], x1[swap] = x1[swap], x0[swap]

    # these are invariant to swapping
    xp = 0.5*(x0+x1)
    fp = 0.5*(f0+f1)

    # loop until convergence or max_iter
    for i in range(max_iter):
        sel0 = (fp <= 0.0)
        sel1 = ~sel0
        x0[sel0] = xp[sel0]
        x1[sel1] = xp[sel1]
        xp = 0.5*(x0+x1)
        fp = fun(xp)
        if np.max(np.abs(fp)) < bs_tol:
            break

    return xp

# golden section search over vector
def goldsec_vec(fun, xmin, xmax, tol=1e-12):
    val0, val1 = fun(xmin), fun(xmax) # testing sizes
    N = len(val0)
    assert(len(val1) == N)

    a = ensure_vector(xmin, N)
    b = ensure_vector(xmax, N)

    c = b - (b-a)/golden
    d = a + (b-a)/golden

    while np.max(np.abs(c-d)) > tol:
        sel = fun(c) < fun(d)
        b[sel] = d[sel]
        a[~sel] = c[~sel]

        c = b - (b-a)/golden
        d = a + (b-a)/golden

    return (b+a)/2

# for an increasing function
def secant_vec(fun, xmin, xmax, max_iter=64, sc_tol=1e-12, output=False):
    x0 = xmin.copy()
    x1 = xmax.copy()
    f0 = fun(x0)
    f1 = fun(x1)
    for i in range(max_iter):
        x0[:] = x1 - f1*((x1-x0)/(f1-f0))
        f0[:] = fun(x0)
        x0, x1 = x1, x0
        f0, f1 = f1, f0
        if np.max(np.abs(f1)) < sc_tol:
            break
    return x1

# stack with flat support
def vstack(v):
    if len(v) == 0:
        return np.array([])
    if v[0].ndim == 1:
        return np.concat(v)
    else:
        return np.vstack(v)

# finite differencing
def deriv(y, x, axis=0, direc='both', pad=True):
    # arrange axes allowing for case of flat x
    if axis != 0:
        y = y.swapaxes(0, axis)
        if x.ndim > 1:
            x = x.swapaxes(0, axis)

    # expand x if it is flat
    if y.ndim > x.ndim:
        x = expand(x, dims=y.ndim-x.ndim)
        # perform differencing

    if direc == 'both':
        der = (y[2:,...]-y[:-2,...])/(x[2:,...]-x[:-2,...])
    elif direc in ('left', 'right'):
        der = (y[1:,...]-y[:-1,...])/(x[1:,...]-x[:-1,...])

    # optionally pad edges
    if pad:
        if direc == 'both':
            der = vstack([der[[0],...], der, der[[-1],...]])
        elif direc == 'left':
            der = vstack([der[[0],...], der])
        elif direc == 'right':
            der = vstack([der, der[[-1],...]])

    # revert axes swap
    if axis != 0:
        der = der.swapaxes(0, axis)

    return der

# generate continuous rv by linearly interpolating using cmf approximations
def random_vec(probs, nf, state=np.random):
    nbins, = probs.shape
    assert((np.log2(nbins)%1.0)==0.0) # only powers of two

    x = state.rand(nf)
    bstep = nbins >> 1
    bpos = np.zeros(nf, dtype=np.int)
    while bstep > 0:
        bcmp = bpos + bstep - 1
        bsel = x > probs[bcmp]
        bpos[bsel] += bstep
        bstep >>= 1

    return bpos

# generate continuous rv by linearly interpolating using cmf approximations (columns of probs)
def random_mat(probs, types, state=np.random):
    ntypes, nbins = probs.shape
    nf, = types.shape
    assert((np.log2(nbins)%1.0)==0.0) # only powers of two

    x = state.rand(nf)
    bstep = nbins >> 1
    bpos = np.zeros(nf, dtype=np.int)
    while bstep > 0:
        bcmp = bpos + bstep - 1
        bsel = x > probs[types, bcmp]
        bpos[bsel] += bstep
        bstep >>= 1

    return bpos
