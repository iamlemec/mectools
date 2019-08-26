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

##
## reading common data sources
##

def world_bank(data, reshape=True, label='value'):
    if type(data) is str:
        data = pd.read_excel(data, sheet_name='Data', skiprows=3)

    # same as PWT
    data = data.rename({'Country Code': 'countrycode'}, axis=1)
    data = data.set_index('countrycode')

    # filter out non-year columns
    data = data.filter(regex=r'\d\d\d\d')
    data.columns = data.columns.astype(np.int)
    data = data.rename_axis('year', axis=1)

    # reshape from wide to long
    if reshape:
        data = data.stack()
        data = data.rename(label)
        data = data.reset_index()

    return data
