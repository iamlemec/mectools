# general vector tools

import json
import toml
import numpy as np
import copy
import operator
import pandas as pd
import collections

# dictionary
class Bundle(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for d in args + (kwargs,):
            self.update(d)

    @classmethod
    def from_subset(cls, d, keys=[]):
        return cls([(k, d[k]) for k in keys])

    @classmethod
    def from_tree(cls, tree):
        if isinstance(tree, dict):
            return cls([(k, cls.from_tree(v)) for k, v in tree.items()])
        else:
            return tree

    @classmethod
    def from_json(cls, path):
        with open(path) as fid:
            return cls.from_tree(json.load(fid))

    @classmethod
    def from_toml(cls, path):
        return cls.from_tree(toml.load(path))

    def __repr__(self):
        return '\n'.join([f'{k} = {v}' for k, v in self.items()])

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def _html_repr_(self):
        rows = '</tr><tr>'.join([f'<td>{k}</td><td>{v}</td>' for k, v in self.items()])
        return f'<table><tr><td>Name</td><td>Value</td></tr><tr>{rows}</tr></table>'

    def keys(self):
        return sorted(super().keys())

    def items(self):
        return sorted(super().items(), key=operator.itemgetter(0))

    def values(self):
        return [v for _, v in self.items()]

    def subset(self, keys):
        return Bundle([(k, v) for k, v in self.items() if k in keys])

    def drop(self, keys):
        return Bundle([(k, v) for k, v in self.items() if k not in keys])

    def apply(self, f):
        return Bundle([(k, f(v)) for k, v in self.items()])

    def to_dict(self):
        return dict(self.items())

    def to_ordered_dict(self):
        return collections.OrderedDict(self.items())

    def to_dataframe(self):
        return pd.DataFrame(self)

    def to_series(self):
        return pd.Series(self)

    def to_array(self):
        return np.array(self.values())

    def to_json(self, path=None, **kwargs):
        kwargs['indent'] = 4
        od = self.to_ordered_dict()
        if path is not None:
            with open(path, 'w+') as fid:
                json.dump(od, fid, **kwargs)
        else:
            return json.dumps(od, **kwargs)

    def copy(self):
        return Bundle(self)

    def deepcopy(self):
        return Bundle(copy.deepcopy(self))

def stack_bundles(bunds, agg_func=np.array, default=np.nan):
    keys = set(sum([b.keys() for b in bunds], []))
    return Bundle({
        (k, agg_func([b.get(k, default) for b in bunds])) for k in keys
    })
