# misc data tools

import os
import operator as op
import collections as co
import itertools as it
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as mpl
import matplotlib.cm as cm
import seaborn as sns
from statsmodels.regression.linear_model import RegressionResultsWrapper

##
## statistics
##

def noinf(s):
    s[np.isinf(s)] = np.nan
    return s

def nonan(s):
    return s[~np.isnan(s)]

def log(s):
    return noinf(np.log(s))

def winsorize(s, level=0.05):
    if type(level) not in (list, tuple):
        level = level, 1.0-level

    clipper = lambda s1: s1.clip(lower=s1.quantile(level[0]), upper=s1.quantile(level[1]))
    if type(s) is pd.Series:
        return clipper(s)
    else:
        return s.apply(clipper)

def mean(s, winsor=False):
    if winsor: s = winsorize(s)
    return s.mean()

def std(s, winsor=False):
    if winsor: s = winsorize(s)
    return s.std()

def iqr(s, level=0.25):
    return s.quantile(0.5+level) - s.quantile(0.5-level)

def quantile(s, p):
    if type(s) is pd.Series:
        return s.quantile(p)
    else:
        return stats.scoreatpercentile(s, 100*p)

def digitize(x, bins):
    if len(x) == 0: return np.array([], dtype=np.int)
    return np.digitize(x, bins)

def vrange(vec, squeeze=0.0):
    if type(squeeze) in (list, tuple):
        return quantile(vec, squeeze[0]), quantile(vec, squeeze[1])
    else:
        return quantile(vec, squeeze), quantile(vec, 1.0-squeeze)

def corr_robust(df, xcol, ycol, wcol=None, winsor=None):
    if xcol == ycol:
        return 1.0, 0.0

    all_vars = [xcol, ycol]
    if wcol is not None:
        all_vars += [wcol]
    df1 = df[all_vars].dropna()

    if winsor is not None:
        df1[[xcol, ycol]] = winsorize(df1[[xcol, ycol]], level=winsor)

    if wcol is None:
        return stats.pearsonr(df1[xcol], df1[ycol])
    else:
        wgt = df1[wcol].astype(np.float)/np.sum(df1[wcol])
        xmean = np.sum(wgt*df1[xcol])
        ymean = np.sum(wgt*df1[ycol])
        xvar = np.sum(wgt*(df1[xcol]-xmean)**2)
        yvar = np.sum(wgt*(df1[ycol]-ymean)**2)
        vcov = np.sum(wgt*(df1[xcol]-xmean)*(df1[ycol]-ymean))
        cval = vcov/np.sqrt(xvar*yvar)
        nval = len(df1)
        tval = cval*np.sqrt((nval-2)/(1.0-cval*cval))
        pval = 2.0*stats.t.sf(np.abs(tval), nval)
        return cval, pval

def gini(x):
    N = len(x)
    B = np.sum(np.arange(N, 0, -1)*np.sort(x))/(N*np.sum(x))
    return 1 + 1/N - 2*B

##
## data frame tools
##

def stack_frames(dfs, prefixes=None, suffixes=None):
    if prefixes is None:
        prefixes = len(dfs)*['']
    elif type(prefixes) not in (tuple,list):
        prefixes = len(dfs)*[prefixes]
    if postfixes is None:
        postfixes = len(dfs)*['']
    elif type(postfixes) not in (tuple,list):
        postfixes = len(dfs)*[postfixes]
    return pd.concat([df.add_prefix(pre).add_suffix(post) for df, pre, post in zip(dfs, prefixes, postfixes)], axis=1)

def compact_out(df, min_col_width=10, col_spacing=1):
    col_spacer = ' '*col_spacing
    row_name_width = max(min_col_width, max(map(lambda x: len(str(x)), df.index)))
    col_widths = map(lambda x: max(min_col_width, len(str(x))), df.columns)
    header_fmt = '{:' + str(row_name_width) + 's}' + col_spacer + '{:>' + ('s}'+col_spacer+'{:>').join(map(str,col_widths)) + '}'
    row_fmt = '{:' + str(row_name_width) + 's}' + col_spacer + '{: ' + ('f}'+col_spacer+'{: ').join(map(str, col_widths)) + 'f}'
    print(header_fmt.format('', *df.columns))
    for i, vs in df.iterrows():
        print(row_fmt.format(str(i), *vs.values))

##
## indexing
##

# this will fill interior values of an index that has a canonical ordering
def gen_range(imin, imax, succ):
    i = imin
    while True:
        if i > imax:
            return
        yield i
        i = succ(i)

def fill_index(sdf, imin=None, imax=None, succ=None, fill_value=np.nan):
    idx = sdf.index
    if imin is None:
        imin = idx.min()
    if imax is None:
        imax = idx.max()
    if succ is None:
        rng = np.arange(imin, imax + 1)
    else:
        rng = gen_range(imin, imax, succ)
    return sdf.reindex(rng, fill_value=fill_value)

##
## variable summary
##

def var_info(datf, var=''):
    if type(datf) is pd.Series:
        svar = datf
    elif type(datf) is pd.DataFrame:
        svar = datf[var]
    print(svar.describe())
    svar.hist()

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
    corr, corr_pval = corr_robust(datf_sel, x_var, y_var, wcol=w_var)

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

def datf_eval(datf, formula, use_numpy=True):
    if use_numpy:
        globs = {'np':np}
    return eval(formula, globs, datf)

##
## regression tools
##

def clustered_covmat(ret, ids1, ids2=None):
    uids1 = np.unique(ids1)
    if ids2 is None:
        errtype = 'clustered'
    else:
        errtype = 'dyadic'
        uids2 = np.unique(ids2)

    u = ret.resid
    X = ret.model.data.exog
    N, K = X.shape

    if errtype == 'clustered':
        B = np.zeros((K, K))
        for i1 in uids1:
            sel = (ids1 == i1)
            Xui = np.dot(X[sel,:].T, u[sel,None])
            B += np.dot(Xui, Xui.T)
        XX1 = np.linalg.inv(np.dot(X.T, X))
        V = np.dot(np.dot(XX1, B), XX1)
    elif errtype == 'dyadic':
        B = np.zeros((K, K))
        for i1 in uids1:
            sel = (ids1 == i1)
            Xui = np.dot(X[sel,:].T, u[sel,None])
            B += np.dot(Xui, Xui.T)
        for i2 in uids2:
            sel = (ids2 == i2)
            Xui = np.dot(X[sel,:].T, u[sel,None])
            B += np.dot(Xui, Xui.T)
        for i1, i2 in zip(uids1, uids2):
            sel = (ids1 == i1) & (ids2 == i2)
            Xui = np.dot(X[sel,:].T, u[sel,None])
            B -= np.dot(Xui, Xui.T)
        XX1 = np.linalg.inv(np.dot(X.T, X))
        V = np.dot(np.dot(XX1, B), XX1)

    # significance stats
    params = ret.params.values
    paridx = ret.params.index
    covmat = pd.DataFrame(V, index=paridx, columns=paridx)
    bse = np.sqrt(np.diag(V))
    tvalues = params/bse
    pvalues = stats.norm.sf(np.abs(tvalues))*2

    # patch and return
    ret.cov_params = lambda: covmat
    ret.pvalues = pvalues
    return ret

def clustered_regression(datf, reg, cid):
    ret = smf.ols(reg, data=datf).fit()
    return clustered_covmat(ret, datf[cid].values)

def dyadic_regression(datf, reg, cid1, cid2):
    ret = smf.ols(reg, data=datf).fit()
    return clustered_covmat(ret, datf[cid1].values, datf[cid2].values)
