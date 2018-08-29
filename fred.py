# FRED tools

import os
import json
import numpy as np
import pandas as pd
from fredapi import Fred

auth_name = 'fred_auth.txt'
auth_path = os.path.join(os.path.dirname(__file__), auth_name)
with open(auth_path) as fa:
    fred_auth = json.load(fa)
fred = Fred(api_key=fred_auth['key'])

cache_dir = os.path.join(os.getenv('HOME'), '.cache/fred_cache')

def get_fred(sid,use_cache=True):
    if use_cache:
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
    fname = os.path.join(cache_dir, '%s.json' % sid)

    # download or use cache
    try:
        assert(use_cache)
        datf = pd.read_csv(fname, index_col=0)
        print('Loaded from cache')
    except:
        print('Fetching from FRED')
        datf = pd.DataFrame({sid: fred.get_series(sid)}).rename_axis('date')
        datf.to_csv(fname)

    # parse index
    ser = datf[sid]
    ser.index = pd.to_datetime(ser.index)

    return ser
