# memory tools

from sys import getsizeof
from types import ModuleType, FunctionType, BuiltinFunctionType
import numpy as np
import pandas as pd

ndrop = ['In', 'Out', 'get_ipython', 'quit', 'exit', 'ndrop', 'tdrop', 'mstr']
tdrop = [type, ModuleType, FunctionType, BuiltinFunctionType]
mstr = ['b', 'k', 'm', 'g', 't', 'p'] # year 2100 compliant

def size_desc(x):
    lvl = 0
    while x >= 1024:
        lvl += 1
        x >>= 10
    return f'{x}{mstr[lvl]}'

def size_fast(x):
    t = type(x)
    if t is pd.DataFrame:
        return x.memory_usage().sum()
    elif t is pd.Series:
        return x.memory_usage()
    elif t is np.ndarray:
        return x.nbytes
    else:
        return getsizeof(x)

def usage(d, sort=True, fast=True, human=True, total=True, pare=True):
    size = size_fast if fast else getsizeof
    if pare:
        d = {k: v for k, v in d.items() if type(v) not in tdrop and k not in ndrop and not k.startswith('_')}
    s = pd.Series({k: size(v) for k, v in d.items()})
    if total:
        s['TOTAL'] = s.sum()
    if sort:
        s = s.sort_values(ascending=False)
    if human:
        s = s.map(size_desc)
    return s
