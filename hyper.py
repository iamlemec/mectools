import re
from itertools import islice

# print iterator progress
def progress(it, per=100, limit=None, fmt='%s'):
    print(fmt % 'starting')
    i = 0
    for x in it:
        yield x
        i += 1
        if i % per == 0:
            print(fmt % str(i))
        if limit is not None and i >= limit:
            break
    print(fmt % 'done')

def iprogress(it, per=100, fmt='%s'):
    print(fmt % 'starting')
    i = 0
    for x in it:
        yield i, x
        i += 1
        if i % per == 0:
            print(fmt % str(i))
    print(fmt % 'done')

# generator of chunks
def chunks(it, size=100):
    itr = iter(it)
    while True:
        x = list(islice(itr, size))
        if len(x) == 0:
            return
        yield x

# generate chunk indices
def ichunks(n, size=100):
    i = 0
    while i < n:
        yield i, i + min(size, n - i)
        i += size

# merge dictionaries
def merge(*ds, **kw):
    ret = {}
    for d in ds + (kw,):
        for k, v in d.items():
            ret[k] = v
    return ret

# file sorting tool
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def natural_sort(text):
    return sorted(text, key=natural_keys)
