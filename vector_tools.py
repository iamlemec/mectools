# agg_risk tools

import json
import numpy as np
import scipy.interpolate as interp
import copy
import operator
import pandas as pd

# dictionary
class Bundle(object):
  def __init__(self,d0={},sub=None):
    self.update(d0,sub=sub)

  def __repr__(self):
    return '\n'.join([str(k)+' = '+str(v) for (k,v) in sorted(self.__dict__.items(),key=operator.itemgetter(0))])

  def _html_repr_(self):
    return '<table><tr><td>Name</td><td>Value</td></tr><tr>'+'</tr><tr>'.join(['<td>{}</td><td>{}</td>'.format(k,v) for (k,v) in self.__dict__.items()])+'</tr></table>'

  def __iter__(self):
    return iter(self.__dict__.items())

  def update(self,d,sub=None):
    if sub is None: sub = d.keys()
    for k in sub:
      setattr(self,k,d[k])

  def items(self):
    return self.__dict__.items()

  def keys(self):
    return self.__dict__.keys()

  def __getitem__(self,name):
    if type(name) is list:
      return Bundle(self.__dict__,sub=name)
    else:
      return self.__dict__[name]

  def __setitem__(self,name,value):
    setattr(self,name,value)

  def dict(self):
    return self.__dict__

  def subset(self,sub):
    return Bundle(self,sub=sub)

  def drop(self,sub):
    return Bundle(self,sub=list(set(self.keys())-set(sub)))

  def to_dataframe(self,sub=None):
    if sub is None: sub = self.__dict__.keys()
    return pd.DataFrame(self.subset(sub).dict())

  def to_series(self,sub=None):
    if sub is None: sub = self.__dict__.keys()
    return pd.Series(self.subset(sub).dict())

  def copy(self):
    return Bundle(self)

  def deepcopy(self):
    b = Bundle(copy.deepcopy(self.__dict__))

# filter dict keys
def filter(d,keys):
  return {k:d[k] for k in keys}

# must have same keys
def stack_bundles(bund_vec,agg_func=np.concatenate):
  return Bundle({k:agg_func([b[k] for b in bund_vec]) for k in bund_vec[0].keys()})

# param set tools
def load_json(fname):
  return json.load(open(fname))

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
bs_tol = 1e-12
def binsearch_vec(fun,xmin,xmax,max_iter=64,output=False):
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
    if np.max(np.abs(fp)) < bs_tol: break
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
  bstep = nbins/2
  bpos = np.zeros(nf,dtype=np.int)
  while bstep >= 1:
    bcmp = bpos + bstep
    bpos[x>probs[bcmp,range(nf)]] += bstep
    bstep /= 2

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
  bstep = nbins/2
  bpos = np.zeros(nf,dtype=np.int)
  while bstep >= 1:
    bcmp = bpos + bstep
    bpos[x>probs[bcmp]] += bstep
    bstep /= 2

  if interp:
    xbin = x*nbins-bpos
    return (1.0-xbin)*vals[bpos] + xbin*vals[bpos+1]
  else:
    return vals[bpos]

def find(v):
  return np.nonzero(v)[0]

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
