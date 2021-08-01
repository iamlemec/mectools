# plotting tools for lecture

import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from . import data as dt

# defaults
figsize0 = 5, 4

# neon colors
neon_blue = '#1e88e5'
neon_red = '#ff0d57'
neon_green = '#13b755'
neon_purple = '#8172b2'

def save_plot(yvars=None, xvar='index', data=None, title=None, labels=None, xlabel=None, ylabel=None, legend=True, xlim=None, ylim=None, figsize=figsize0, tight=True):
    if yvars is None:
        yvars = list(data.columns)
    if type(yvars) is np.ndarray:
        if yvars.ndim == 1:
            yvars = [yvars]
        else:
            yvars = list(yvars)
    if type(yvars) is not list:
        yvars = [yvars]
    if data is not None:
        if legend is True:
            legend = yvars
        if xvar == 'index':
            xvar = data.index.values
        else:
            xvar = data[xvar].values
    else:
        if xvar == 'index':
            xvar = np.arange(len(yvars[0]))

    fig, ax = plt.subplots(figsize=figsize)
    if data is not None:
        data[yvars].plot(ax=ax, legend=legend)
    else:
        for yvar in yvars:
            ax.plot(xvar, yvar)

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if tight: plt.axis('tight')
    if legend: ax.legend(legend, loc='best')

    return fig

def scatter_label(xvar, yvar, data, labels=None, alpha=None, offset=0.02, figsize=figsize0, ax=None):
    cols = [xvar, yvar]
    if labels is not None:
        cols += [labels]
    if alpha is not None:
        cols += [alpha]
    data = data[cols].dropna().copy()

    if labels is not None:
        data = data.set_index(labels)
    if alpha is None:
        alpha = 'alpha'
        data['alpha'] = 1

    if ax is None: _, ax = plt.subplots(figsize=figsize)
    data.plot.scatter(x=xvar, y=yvar, s=0, ax=ax)

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ydel, xdel = ymax - ymin, xmax - xmin

    for txt, row in data.iterrows():
        pos = row[xvar] - 0.02*xdel, row[yvar] - 0.02*ydel
        ax.annotate(txt, pos, alpha=row[alpha])

    return ax

def v_range(s, wins):
    s1 = dt.winsorize(s, level=wins)
    return s1.min(), s1.max()

# pretty flexible dataframe correlation plotter
def corr_info(datf, x_var, y_var, w_var=None, c_var='index', ax=None,
              x_range=None, y_range=None, x_name=None, y_name=None, title='',
              reg_type=None, size_scale=1.0, winsor=None, graph_squeeze=0.05,
              alpha=0.8, color_skew=0.5, style=None, palette=None,
              grid=None, logx=False, logy=False, loglog=False):
    # shallow copy
    datf = datf.copy()

    # sort out needed columns
    all_vars = [x_var, y_var]
    if w_var:
        all_vars += [w_var]
    if c_var and not c_var == 'index':
        all_vars += [c_var]

    # transformations
    if logx or loglog:
        datf[x_var] = log(datf[x_var])
    if logy or loglog:
        datf[y_var] = log(datf[y_var])

    # select proper rows
    datf_sel = datf[all_vars].dropna()
    if winsor is not None:
        datf_sel[[x_var, y_var]] = winsorize(datf_sel[[x_var, y_var]], level=winsor)

    # regression select
    if reg_type is None:
        if w_var is None:
            reg_type = 'OLS'
        else:
            reg_type = 'WLS'

    # axes labels
    if x_name is None: x_name = x_var
    if y_name is None: y_name = y_var

    # axes ranges
    if x_range is None: x_range = v_range(datf_sel[x_var], graph_squeeze)
    if y_range is None: y_range = v_range(datf_sel[y_var], graph_squeeze)

    # regression data
    datf_regy = datf_sel[y_var]
    datf_regx = sm.add_constant(datf_sel[[x_var]])

    x_vals = np.linspace(x_range[0], x_range[1], 128)
    xp_vals = sm.add_constant(x_vals)

    # execute regression
    if reg_type == 'WLS':
        mod = sm.WLS(datf_regy, datf_regx, weights=datf_sel[w_var])
        res = mod.fit()
        y_vals = res.predict(xp_vals)
    elif reg_type == 'kernel':
        mod = sm.nonparametric.KernelReg(datf_regy, datf_regx[x_var], 'c')
        (y_vals, _) = mod.fit(x_vals)
    else:
        reg_unit = getattr(sm, reg_type)
        mod = reg_unit(datf_regy, datf_regx)
        res = mod.fit()
        y_vals = res.predict(xp_vals)

    # raw correlations
    corr, corr_pval = dt.corr_robust(datf_sel, x_var, y_var, wcol=w_var)

    # display regression results
    if reg_type != 'kernel':
        str_width = max(11,len(x_var))
        fmt_0 = '{:'+str(str_width)+'s} = {: f}'
        fmt_1 = '{:'+str(str_width)+'s} = {: f} ({:f})'
        print(fmt_0.format('constant',res.params['const']))
        print(fmt_1.format(x_var,res.params[x_var],res.pvalues[x_var]))
        print(fmt_1.format('correlation',corr,corr_pval))
        #print(fmt_0.format('R-squared',res.rsquared))

    # figure out point size
    if w_var:
        wgt_norm = datf_sel[w_var]
        wgt_norm /= np.mean(wgt_norm)
    else:
        wgt_norm = np.ones_like(datf_sel[x_var])

    # calculate point color
    if c_var is not None:
        if c_var == 'index':
            idx_norm = datf_sel.index.values.astype(np.float)
        else:
            idx_norm = datf_sel[c_var].values.astype(np.float)
    else:
        idx_norm = np.ones(len(datf_sel))
    idx_norm -= np.mean(idx_norm)
    idx_std = np.std(idx_norm)
    if idx_std > 0.0:
        idx_norm /= 2.0*np.std(idx_norm)
    idx_norm = color_skew*idx_norm + 1.0
    color_args = 0.1 + 0.6*idx_norm
    color_vals = cm.Blues(color_args)

    # set plot style
    if style: sns.set_style(style)
    if palette: sns.set_palette(palette)

    # construct axes
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(7,5))

    # execute plots
    ax.scatter(datf_sel[x_var], datf_sel[y_var], s=20.0*size_scale*wgt_norm, color=color_vals, alpha=alpha)
    ax.plot(x_vals, y_vals, color='r', linewidth=1.0, alpha=0.7)

    # custom plot styling
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    if grid: ax.grid(grid)

    # return
    if reg_type != 'kernel':
        return ax, res
    else:
        return ax

def grid_plots(eqvars, x_vars, y_vars, shape, x_names=None, y_names=None,
               x_ranges=None, y_ranges=None, legends=None, legend_locs=None,
               figsize=(5, 4.5), fontsize=None, file_name=None,
               show_graphs=True, pcmd='plot', extra_args={}):
    import matplotlib.pyplot as plt

    if pcmd == 'bar':
        color_cycle = mpl.rcParams['axes.prop_cycle']
        def pfun(ax, x_data, y_data, **kwargs):
            n_series = len(y_data.T)
            tot_width = np.ptp(x_data)
            width = float(tot_width)/len(x_data)/n_series
            for i, (cc, y_series) in enumerate(zip(color_cycle, y_data.T)):
                ax.bar(x_data+(i-float(n_series)/2)*width, y_series, width, **dict(cc, **kwargs))
    else:
        def pfun(ax, x_data, y_data, **kwargs):
            getattr(ax, pcmd)(x_data, y_data, **kwargs)

    n_plots = len(y_vars)
    if y_names is None:
        y_names = n_plots*[None]
    if y_ranges is None:
        y_ranges = n_plots*[None]
    if legends is None:
        legends = n_plots*[None]
    if legend_locs is None:
        legend_locs = n_plots*[None]

    if type(x_vars) is not list:
        x_vars = n_plots*[x_vars]
    if type(x_names) is not list:
        x_names = n_plots*[x_names]
    if type(x_ranges) is not list:
        x_ranges = n_plots*[x_ranges]
    if type(extra_args) is not list:
        extra_args = n_plots*[extra_args]

    rows, cols = shape
    figx0, figy0 = figsize
    figx, figy = figx0*cols, figy0*rows
    fig, axlist = plt.subplots(rows, cols, figsize=(figx, figy))
    axlist = axlist.flatten()[:n_plots]
    for xv, yv, xn, yn, xr, yr, lg, ll, ax, ea in zip(x_vars, y_vars, x_names, y_names, x_ranges, y_ranges, legends, legend_locs, axlist, extra_args):
        x_data = eqvars[xv]
        if type(yv) is not list:
            yv = [yv]
        y_data = np.array([eqvars[yv1] for yv1 in yv]).T
        pfun(ax, x_data, y_data, **ea)
        ax.locator_params(nbins=7)
        if xn is not None:
            ax.set_xlabel(xn)
        if xr is not None:
            ax.set_xlim(xr)
        if yn is not None:
            ax.set_title(yn)
        if yr is not None:
            ax.set_ylim(yr)
        if lg is not None:
            ax.legend(lg, loc=[ll if ll is not None else 'best'])

    fig.subplots_adjust(bottom=0.15)
    fig.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)

    if show_graphs:
        plt.show()
    else:
        plt.close()

##
## distributions over paths
##

def path_dist(
    df, ax=None, kind='shade', median=True, mean=True, wins=False, wgt=None,
    quants=None, qmin=0.05, qmax=0.45, alpha=None, figsize=None, color1=neon_blue,
    color2=neon_red
):
    if type(df) is np.ndarray:
        df = pd.DataFrame(df)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if wgt is None:
        wgt = np.ones(df.shape[1])
    elif type(wgt) is str:
        wgt = df[wgt]

    if quants is None:
        quants = 25
    if type(quants) is int:
        quants = np.linspace(qmin, qmax, quants)

    if alpha is None:
        if kind == 'shade':
            alpha = 1.5/len(quants)
        elif kind == 'wisp':
            alpha = 0.1

    if median:
        df.median(axis=1).plot(ax=ax, c=color1)

    if mean:
        if wins:
            vmin, vmax = df.quantile(qmin, axis=1), df.quantile(qmax, axis=1)
            df1 = df.clip(vmin, vmax, axis=0)
        else:
            df1 = df
        df1.mul(wgt, axis=1).sum(axis=1).div((df1.notna()*wgt).sum(axis=1)).plot(ax=ax, c=color2)

    if kind == 'shade':
        for x in quants:
            qlo = df.quantile(0.5-x, axis=1)
            qhi = df.quantile(0.5+x, axis=1)
            ax.fill_between(df.index, qlo, qhi, alpha=alpha, color=color1)
    elif kind == 'wisp':
        n = len(df.columns)
        per = max(1, n // 100)
        df.loc[:, ::per].plot(c=color1, alpha=alpha, ax=ax, legend=False)

    return ax

def kdeplot(df, ax=None, clip=None, wins=0.001, nbins=100, bw=None, fill=True, alpha=0.3, color1=neon_blue, color2=neon_red):
    if ax is None:
        _, ax = plt.subplots()

    # infs to nans
    df = dt.noinf(df)
    if type(df) is np.ndarray:
        df = pd.Series(df)
    if type(df) is pd.Series:
        df = pd.DataFrame(df)

    # find grid
    if clip is None:
        if type(wins) is not tuple:
            winsl, winsr = wins, wins
        else:
            winsl, winsr = wins
        vmin = df.quantile(winsl).min()
        vmax = df.quantile(1-winsr).max()
    else:
        vmin, vmax = clip
    grid = np.linspace(vmin, vmax, nbins)

    # compute kde and plot
    for s in df:
        dat = df[s].dropna().clip(vmin, vmax)
        kde = stats.gaussian_kde(dat, bw_method=bw)
        pdf = kde(grid)
        ax.plot(grid, pdf, linewidth=1, color=color2)
        if fill:
            ax.fill_between(grid, 0, pdf, alpha=alpha, color=color1)
        ax.set_ylim(bottom=0)
        ax.set_xlim(vmin, vmax)

    return ax

##
## log plots
##

def gen_ticks(ymin, ymax):
    ppow = np.floor(np.log10(ymin))
    pnum = ymin/np.power(10.0, ppow)

    if pnum < 2:
        pnum = 1
    elif pnum < 5:
        pnum = 2
    else:
        pnum = 5

    yval = pnum*(10**ppow)
    while yval <= ymax:
        yield yval

        if pnum == 1:
            pnum = 2
        elif pnum == 2:
            pnum = 5
        else:
            pnum = 1
            ppow += 1

        yval = pnum*(10**ppow)

    yield yval

class FixedLogScale(mpl.scale.ScaleBase):
    name = 'fixed_log'

    def __init__(self, axis, scale=1e6):
        super().__init__(axis)
        self.scale = scale

    def get_transform(self):
        return mpl.scale.FuncTransform(np.log, np.exp)

    def set_default_locators_and_formatters(self, axis):
        class InverseFormatter(mpl.ticker.Formatter):
            def __init__(self, scale):
                self.scale = scale

            def __call__(self, x, pos=None):
                d = self.scale*x
                if d >= 1:
                    return '%d' % int(d)
                else:
                    return '%.1f' % d

        ymin, ymax = axis.get_view_interval()
        ymin = np.maximum(1/self.scale, ymin)
        ticks = list(gen_ticks(ymin, ymax))
        loc = mpl.ticker.FixedLocator(ticks)

        axis.set_major_locator(loc)
        axis.set_major_formatter(InverseFormatter(self.scale))
        axis.set_minor_locator(mpl.ticker.NullLocator())
mpl.scale.register_scale(FixedLogScale)
