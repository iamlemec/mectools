# general vector tools

import json
import numpy as np
import scipy.interpolate as interp
import copy
import operator
import pandas as pd
import collections

# dictionary
class Bundle(object):
    def __init__(self,d0={},sub=None):
        self.update(d0,sub=sub)

    def __repr__(self):
        return '\n'.join([str(k)+' = '+str(v) for (k,v) in sorted(self.__dict__.items(),key=operator.itemgetter(0))])

    def _html_repr_(self):
        return '<table><tr><td>Name</td><td>Value</td></tr><tr>'+'</tr><tr>'.join(['<td>{}</td><td>{}</td>'.format(k,v) for (k,v) in self.__dict__.items()])+'</tr></table>'

    def __iter__(self):
        return iter(self.__dict__.keys())

    def update(self,d,sub=None):
        if sub is None: sub = d.keys()
        for k in sub:
            setattr(self,k,d[k])
        return self

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __getitem__(self,name):
        if type(name) is list:
            return Bundle(self.__dict__,sub=name)
        else:
            return self.__dict__[name]

    def __setitem__(self,name,value):
        setattr(self,name,value)

    def dict(self):
        return self.__dict__

    def get(self,key,de=None):
        return self.__dict__.get(key,de)

    def subset(self,sub):
        return Bundle(self,sub=sub)

    def drop(self,sub):
        return Bundle(self,sub=list(set(self.keys())-set(sub)))

    def apply(self,f):
        return Bundle({k:f(v) for (k,v) in self.__dict__.items()})

    def to_dataframe(self):
        return pd.DataFrame(self.__dict__)

    def to_series(self):
        return pd.Series(self.__dict__)

    def to_array(self):
        return np.array(self.values()).transpose()

    def to_json(self,file_name=None,**kwargs):
        kwargs['indent'] = 4
        od = collections.OrderedDict(sorted(self.items()))
        if file_name is not None:
            json.dump(od,open(file_name,'w+'),**kwargs)
        else:
            return json.dumps(od,**kwargs)

    def copy(self):
        return Bundle(self)

    def deepcopy(self):
        return Bundle(copy.deepcopy(self.__dict__))

def bundle(**kwargs):
    return Bundle(kwargs)

# filter dict keys
def filter(d,keys):
    return {k:d[k] for k in keys}

# map with dict/func
def map(d,keys):
    if type(d) is dict:
        return [d[k] for k in keys]
    else:
        return map(d,keys)

# must have same keys
def stack_bundles(bund_vec,agg_func=np.array):
    return Bundle({k:agg_func([b[k] for b in bund_vec]) for k in bund_vec[0].keys()})

# param set tools
def load_json(fname,ordered=False):
    if ordered:
        return json.load(open(fname),object_pairs_hook=collections.OrderedDict)
    else:
        return json.load(open(fname))

def save_json(d,fname):
    json.dump(d,open(fname,'w+'),indent=4)

def bundle_recurse(d):
    b = Bundle(d)
    for k in b.keys():
        if type(b[k]) is dict:
            b[k] = Bundle(b[k])
    return b

def bundle_json(fname):
    return bundle_recurse(load_json(fname))

def local_dict(vnames,loc):
    return dict([(vn,loc[vn]) for vn in vnames])

def local_bundle(vnames,loc):
    return Bundle(local_dict(vnames,loc))

# valfunc utils
def disc_diff_d1(vec,size,width):
    rvec = vec.reshape(size)
    dvec = np.zeros(size)
    dvec[:,0] = (rvec[:,1]-rvec[:,0])/width
    dvec[:,-1] = (rvec[:,-1]-rvec[:,-2])/width
    dvec[:,1:-1] = (rvec[:,2:]-rvec[:,:-2])/(2.0*width)
    return dvec.flatten()

def disc_diff_d2(vec,size,width):
    rvec = vec.reshape(size)
    dvec = np.zeros(size)
    dvec[0,:] = (rvec[1,:]-rvec[0,:])/width
    dvec[-1,:] = (rvec[-1,:]-rvec[-2,:])/width
    dvec[1:-1,:] = (rvec[2:,:]-rvec[:-2,:])/(2.0*width)
    return dvec.flatten()

def disc_diff2_d1(vec,size,width):
    rvec = vec.reshape(size)
    dvec = np.zeros(size)
    dvec[:,1:-1] = ((rvec[:,2:]-rvec[:,1:-1])/width-(rvec[:,1:-1]-rvec[:,:-2])/width)/width
    dvec[:,0] = dvec[:,1]
    dvec[:,-1] = dvec[:,-2]
    return dvec.flatten()

# for an increasing function
def secant_vec(fun,xmin,xmax,max_iter=64,sc_tol=1e-12,output=False):
    """Find zeros of a vector of increasing functions."""
    x0 = xmin.copy()
    x1 = xmax.copy()
    f0 = fun(x0)
    f1 = fun(x1)
    for i in range(max_iter):
        x0[:] = x1 - f1*((x1-x0)/(f1-f0))
        f0[:] = fun(x0)
        (x0,x1) = (x1,x0)
        (f0,f1) = (f1,f0)
        if np.max(np.abs(f1)) < sc_tol:
            break
    return x1

# for an increasing function
def binsearch_vec(fun,xmin,xmax,max_iter=64,bs_tol=1e-12,output=False):
    """Find zeros of a vector of increasing functions."""
    x0 = xmin.copy()
    x1 = xmax.copy()
    xp = 0.5*(x0+x1)
    for i in range(max_iter):
        fp = fun(xp)
        sel0 = (fp <= 0.0)
        sel1 = ~sel0
        x0[sel0] = xp[sel0]
        x1[sel1] = xp[sel1]
        xp = 0.5*(x0+x1)
        if np.max(np.abs(fp)) < bs_tol:
            break
    return xp

# x ~ N_firms , bins ~ N_states x N_firms
# start from top bin
def digitize_mat(x,bins):
    (nbins,nf) = bins.shape
    assert(x.shape==(nf,))
    outv = nbins*np.ones(nf)
    for i in range(nbins-1,-1,-1):
        outv[x<bins[i,:]] = i
    return outv

# generate continuous rv by linearly interpolating using cmf approximations (columns of probs)
# last row should be all ones
def random_panel(probs,vals,interp=False):
    (nbins,nf) = probs.shape
    assert((np.log2(nbins)%1.0)==0.0) # only powers of two
    assert(vals.shape==(nbins,))

    x = np.random.rand(nf)
    bstep = nbins >> 1
    bpos = np.zeros(nf,dtype=np.int)
    while bstep >= 1:
        bcmp = bpos+bstep-1
        bpos[x>probs[bcmp,range(nf)]] += bstep
        bstep >>= 1

    if interp:
        xbin = x*nbins-bpos
        return (1.0-xbin)*vals[bpos] + xbin*vals[bpos+1]
    else:
        return vals[bpos]

# generate continuous rv by linearly interpolating using cmf approximations
# last element should be all ones
def random_vec(probs,vals,nf):
    (nbins,) = probs.shape
    assert((np.log2(nbins)%1.0)==0.0) # only powers of two
    assert(vals.shape==(nbins,))

    x = np.random.rand(nf)
    bstep = nbins >> 1
    bpos = np.zeros(nf,dtype=np.int)
    while bstep >= 1:
        bcmp = bpos + bstep
        bpos[x>probs[bcmp]] += bstep
        bstep >> 1

    if interp:
        xbin = x*nbins-bpos
        return (1.0-xbin)*vals[bpos] + xbin*vals[bpos+1]
    else:
        return vals[bpos]

def find(m):
    return np.nonzero(m)[0]

# def find(m,axis=0):
    # d = len(m.shape)
    # n = m.shape[axis]
    # ret = -np.ones([x for (i,x) in enumerate(m.shape) if i != axis],dtype=np.int)
    # for i in range(n):
        # s = [i if j == axis else slice(None) for j in range(d)]
        # ret[(m[s]!=0)&(ret==-1)] = i
    # return ret

def digitize(x,bins):
    if x.shape == (0,):
        return np.array([],dtype=np.int)
    else:
        return np.digitize(x,bins)

def flatten(v):
    return v.flatten()

class LinearInterpClamp:
    def __init__(self,x,y):
        self.lin_interp = interp.LinearNDInterpolator(x,y)
        self.bnd_interp = interp.NearestNDInterpolator(x,y)

    def __call__(self,*z):
        rvec = self.lin_interp(*z)
        nan_sel = np.isnan(rvec)
        if rvec.ndim > 0:
            rvec[nan_sel] = self.bnd_interp(*[zp[nan_sel] for zp in z])
        elif nan_sel:
            rvec = self.bnd_interp(*z)
        return rvec

# allow to take ndarray inputs
class UnivariateSpline(interp.UnivariateSpline):
    def __call__(self,x,**kwargs):
        if type(x) is np.ndarray and x.ndim > 0:
            return super(UnivariateSpline,self).__call__(x.flat,**kwargs).reshape(x.shape)
        else:
            return super(UnivariateSpline,self).__call__(x,**kwargs)

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def push_index(df,vals):
    df.index = vals
    return df
