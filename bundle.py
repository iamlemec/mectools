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
        return list(self.__dict__.keys())

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

    def to_dataframe(self, index=None):
        return pd.DataFrame(self.__dict__, index=index)

    def to_series(self, index=None):
        return pd.Series(self.__dict__, index=index)

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

# must have same keys
def stack_bundles(bund_vec, agg_func=np.array, default=np.nan):
    keys = set(sum([b.keys() for b in bund_vec], []))
    return Bundle({k: agg_func([b.get(k, default) for b in bund_vec]) for k in keys})

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
    return Bundle(local_dict(vnames, loc))

def load_json(fname, ordered=False):
    if ordered:
        return json.load(open(fname),object_pairs_hook=collections.OrderedDict)
    else:
        return json.load(open(fname))

def save_json(d,fname):
    json.dump(d,open(fname,'w+'),indent=4)
