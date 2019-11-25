import numpy as np
import pandas as pd
import altair as alt
import matplotlib as mpl
import matplotlib.style
from statsmodels.nonparametric.kde import kdensity

# patcher loader for matplotlib
def plotter(pyplot=True, backend='TkAgg', theme=['clean'], rc={}):
    mpl.use(backend)
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

# define the theme by returning the dictionary of configurations
def altair_config(size=14, width=400, height=300, glob=True):
    theme = {
        'config': {
            'view': {
                'height': height,
                'width': width,
                'strokeWidth': 0
            },
            'axis': {
                'grid': False,
                'domainColor': 'black',
                'tickColor': 'black',
                'titleFontSize': size,
                'labelFontSize': size
            },
            'legend': {
                'titleFontSize': size,
                'labelFontSize': size
            }
        }
    }
    if glob:
        import altair as alt
        import pdvega
        # pd.set_option('plotting.backend', 'altair')
        alt.themes.register('mec', lambda: theme)
        alt.themes.enable('mec')
        return alt
    else:
        return theme

def altair_hist(data, log=False, q=0.01, bins=10):
    data = data.dropna()
    bin_lo, bin_hi = data.quantile([q, 1-q])
    if type(bins) is int:
        if log:
            bins = np.exp(np.linspace(np.log(bin_lo), np.log(bin_hi), bins+1))
        else:
            bins = np.linspace(bin_lo, bin_hi, bins+1)
    mids = 0.5*(bins[:-1]+bins[1:])
    count, _ = np.histogram(data, bins=bins)
    hist = pd.DataFrame({'bin': mids, 'count': count})
    hist['dense'] = hist['count']/len(data)
    ch = alt.Chart(hist).mark_bar(size=20).encode(x='bin', y='dense')
    return ch

def altair_kde(data, q=0.01, bins=100):
    data = data.dropna()
    clip = data.quantile([q, 1-q]).values
    kde, bins, bw = kdensity(data, clip=clip, gridsize=bins)
    hist = pd.DataFrame({'bin': bins, 'kde': kde})
    ch = alt.Chart(hist).mark_line().encode(x='bin', y='kde')
    return ch
