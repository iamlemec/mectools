import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.style

# patcher loader for matplotlib
def plotter(backend=None, theme=['clean'], pyplot=True, rc={}):
    if backend is not None:
        mpl.use(backend)
    if theme is not None:
        mpl.style.use(theme)
    mpl.rcParams.update(rc)
    mpl.interactive(True)

    # kill hist grid
    if not hasattr(pd.Series, 'hist0'):
        pd.Series.hist0 = pd.Series.hist
        def hist(*args, **kwargs):
            if 'grid' not in kwargs:
                kwargs['grid'] = False
            return pd.Series.hist0(*args, **kwargs)
        setattr(pd.Series, 'hist', hist)

    if pyplot:
        import matplotlib.pyplot as plt
        return plt
