# patcher loader for matplotlib
def plotter(pyplot=True, backend='GTK3Agg', theme=['clean'], rc={}):
    import matplotlib as mpl
    import matplotlib.style
    import pandas as pd

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
    import altair as alt
    if glob:
        alt.themes.register('mec', lambda: theme)
        alt.themes.enable('mec')
    else:
        return theme
