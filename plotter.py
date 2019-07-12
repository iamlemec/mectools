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
            pd.Series.hist0(*args, **kwargs)
        setattr(pd.Series, 'hist', hist)

    if pyplot:
        import matplotlib.pyplot as plt
        return plt

# define the theme by returning the dictionary of configurations
def altair(size=14):
    def theme():
        return {
            'config': {
                'view': {
                    'height': 300,
                    'width': 400,
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
    alt.themes.register('mec', theme)
    alt.themes.enable('mec')
    return alt
