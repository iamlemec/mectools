# misc data tools

import os
import operator as op
import collections as co
import itertools as it
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.cm as cm
import seaborn as sns
import patsy

# statistics

def noinf(s):
    s[np.isinf(s)] = np.nan
    return s

def nonan(s):
    return s[~np.isnan(s)]

def log(s):
    return noinf(np.log(s))

def winsorize(s,level=0.05):
    if type(level) not in (list,tuple):
        level = (level,1.0-level)

    clipper = lambda s1: s1.clip(lower=s1.quantile(level[0]),upper=s1.quantile(level[1]))
    if type(s) is pd.Series:
        return clipper(s)
    else:
        return s.apply(clipper)

def mean(s,winsor=False):
    if winsor: s = winsorize(s)
    return s.mean()

def std(s,winsor=False):
    if winsor: s = winsorize(s)
    return s.std()

def iqr(s,level=0.25):
    return s.quantile(0.5+level)-s.quantile(0.5-level)

def quantile(s,p):
    if type(s) is pd.Series:
        return s.quantile(p)
    else:
        return stats.scoreatpercentile(s,100.0*p)

def digitize(x,bins):
    if len(x) == 0: return np.array([],dtype=np.int)
    return np.digitize(x,bins)

def v_range(vec,squeeze=0.0):
    if type(squeeze) in (list,tuple):
        return (quantile(vec,squeeze[0]),quantile(vec,squeeze[1]))
    else:
        return (quantile(vec,squeeze),quantile(vec,1.0-squeeze))

def corr_robust(df,xcol,ycol,wcol=None,winsor=None):
    if xcol == ycol: return (1.0,0.0)

    all_vars = [xcol,ycol]
    if wcol is not None: all_vars += [wcol]
    df1 = df[all_vars].dropna()

    if winsor is not None:
        df1[[xcol,ycol]] = winsorize(df1[[xcol,ycol]],level=winsor)

    if wcol is None:
        return stats.pearsonr(df1[xcol],df1[ycol])
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
        pval = 2.0*stats.t.sf(np.abs(tval),nval)
        return (cval,pval)

def gini(x):
    N = len(x)
    B = np.sum(np.arange(N,0,-1)*np.sort(x))/(N*np.sum(x))
    return 1.0+(1.0/N)-2.0*B

# data frame tools

def prefixer(sp):
    return lambda s: sp+s

def postfixer(sp):
    return lambda s: s+sp

def prefix(sv,sp):
    return map(prefixer(sp),sv)

def postfix(sv,sp):
    return map(postfixer(sp),sv)

def stack_frames(dfs,prefixes=None,postfixes=None):
    if prefixes is None:
        prefixes = len(dfs)*['']
    elif type(prefixes) not in (tuple,list):
        prefixes = len(dfs)*[prefixes]
    if postfixes is None:
        postfixes = len(dfs)*['']
    elif type(postfixes) not in (tuple,list):
        postfixes = len(dfs)*[postfixes]
    return pd.concat([df.rename(columns=prefixer(pre)).rename(columns=postfixer(post)) for (df,pre,post) in zip(dfs,prefixes,postfixes)],axis=1)

def compact_out(df,min_col_width=10,col_spacing=1):
    col_spacer = ' '*col_spacing
    row_name_width = max(min_col_width,max(map(lambda x: len(str(x)),df.index)))
    col_widths = map(lambda x: max(min_col_width,len(str(x))),df.columns)
    header_fmt = '{:'+str(row_name_width)+'s}'+col_spacer+'{:>'+('s}'+col_spacer+'{:>').join(map(str,col_widths))+'}'
    row_fmt = '{:'+str(row_name_width)+'s}'+col_spacer+'{: '+('f}'+col_spacer+'{: ').join(map(str,col_widths))+'f}'
    print(header_fmt.format('',*df.columns))
    for (i,vs) in df.iterrows(): print(row_fmt.format(str(i),*vs.values))

# fix rolling_cov in pandas

def rolling_cov(arg1, arg2, window, center=False):
    window = min(window, len(arg1), len(arg2))

    mean = lambda x: pd.rolling_mean(x, window, center=center)
    count = pd.rolling_count(arg1 + arg2, window, center=center)
    bias_adj = count / (count - 1)
    return (mean(arg1 * arg2) - mean(arg1) * mean(arg2)) * bias_adj

def rolling_corr(arg1, arg2, window, center=False):
    num = rolling_cov(arg1, arg2, window, center=center)
    den = pd.rolling_std(arg1, window, center=center) * pd.rolling_std(arg2, window, center=center)
    return num / den

# butterworth filter

import scipy.signal as sig

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    (b,a) = sig.butter(order,[low,high],btype='band')
    return (b,a)

def butter_bandpass_filter(data,lowcut,highcut,fs=1.0,order=5):
    (b,a) = butter_bandpass(lowcut,highcut,fs,order=order)
    y = sig.lfilter(b,a,data)
    return y

# block filters

# this will drop NaNs but doesn't handle non-uniform series
def lowpass_filter(data,cutoff,fs=1.0):
    freqs = np.fft.fftfreq(len(data),d=fs)
    fftv = np.fft.fft(data)
    fftv[np.abs(freqs)>=cutoff] = 0.0
    smooth = np.fft.ifft(fftv)
    return smooth.real

def bandpass_filter(data,lowcut,highcut,fs=1.0):
    freqs = np.fft.fftfreq(len(data),d=fs)
    fftv = np.fft.fft(data)
    cut_sel = (np.abs(freqs)<=lowcut) | (np.abs(freqs)>=highcut)
    cut_sel[0] = False
    fftv[cut_sel] = 0.0
    smooth = np.fft.ifft(fftv)
    return smooth.real

# plotting panels

def plot_filter(datf,cols=None,ftype='lowpass',show_orig=True,fargs=[],fkwargs={},pargs=[],pkwargs={}):
    if isinstance(datf,pd.DataFrame):
        if cols == None: cols = datf.columns
        datf = datf.filter(cols)
    else:
        datf = pd.DataFrame({'value':datf})

    datf = datf.dropna()

    if ftype == 'butter':
        yfunc = lambda s: butter_bandpass_filter(s,*fargs,**fkwargs)
    elif ftype == 'lowpass':
        yfunc = lambda s: lowpass_filter(s,*fargs,**fkwargs)
    elif ftype == 'bandpass':
        yfunc = lambda s: bandpass_filter(s,*fargs,**fkwargs)
    elif ftype == 'smooth':
        yfunc = lambda s: pd.rolling_mean(s,*fargs,**fkwargs)

    datf_filt = datf.apply(yfunc,axis=0).rename(columns=lambda s: s+'_filt')
    ax = datf_filt.plot(linewidth=2.0,grid=True,*pargs,**pkwargs)
    if show_orig:
        datf.plot(ax=ax,linewidth=0.5,grid=True,*pargs,**pkwargs)
        step = len(ax.lines)/2
        for i in range(step): ax.lines[i+step].set_color(ax.lines[i].get_color())

def plot_panel(df,cols=None,detrend=False,norm=False,norm_pos=False,**kwargs):
    if cols == None: cols = df.columns
    df_final = df.filter(cols).copy()

    if detrend:
        df_final = df_final.dropna().apply(sig.detrend)

    if norm:
        df_final = (df_final-df_final.median())/(df_final.quantile(0.75)-df_final.quantile(0.25))
    if norm_pos:
        df_final = df_final/df_final.mean()

    plt.plot(df_final.index,df_final.values,**kwargs)
    plt.legend(cols,loc='best')

def hist_panel(s,clip=None,**kwargs):
    s_final = s.dropna().copy()

    if clip != None:
        s_final = winsorize(s_final,level=clip)

    plt.hist(s_final,**kwargs)

def heatmap(x,y,bins=30,range=None):
    (heatmap,xedges,yedges) = np.histogram2d(-x,y,bins=bins,range=range)
    plt.imshow(heatmap,extent=[-xedges[-1],-xedges[0],yedges[0],yedges[-1]],aspect='auto',interpolation='nearest')
    plt.colorbar()

def heatmap_panel(datf,cols=None,bins=30,range=None):
    if cols is None: cols = datf.columns
    (col1,col2) = cols
    datf = datf.dropna()
    heatmap(datf[col1],datf[col2],bins=bins,range=range)
    plt.xlabel(col1)
    plt.ylabel(col2)

# variable summary

def var_info(datf,var=''):
    if type(datf) is pd.Series:
        svar = datf
    elif type(datf) is pd.DataFrame:
        svar = datf[var]
    print(svar.describe())
    svar.hist()

# pretty flexible dataframe correlation plotter
def corr_info(datf, x_var, y_var, w_var=None, c_var='index', ax=None, x_range=None, y_range=None, x_name=None, y_name=None, title='', reg_type=None, size_scale=1.0, winsor=None, graph_squeeze=0.05, alpha=0.8, color_skew=0.5, style=None, palette=None, despine=None, grid=None, logx=False, logy=False, loglog=False):
    # shallow copy
    datf = datf.copy()

    # sort out needed columns
    all_vars = [x_var, y_var]
    if w_var: all_vars += [w_var]
    if c_var and not c_var == 'index': all_vars += [c_var]

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
    (corr, corr_pval) = corr_robust(datf_sel, x_var, y_var, wcol=w_var)

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
        (fig, ax) = plt.subplots(figsize=(7,5))
    else:
        fig = ax.figure

    # execute plots
    ax.scatter(datf_sel[x_var], datf_sel[y_var], s=20.0*size_scale*wgt_norm, color=color_vals, alpha=alpha)
    ax.plot(x_vals, y_vals, color='r', linewidth=1.0, alpha=0.7)

    # custom plot styling
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    if despine: sns.despine(fig,ax)
    if grid: ax.grid(grid)

    # return
    if reg_type != 'kernel':
        return (fig, ax, res)
    else:
        return (fig, ax)

def grid_plots(eqvars,x_vars,y_vars,shape,x_names=None,y_names=None,x_ranges=None,y_ranges=None,legends=None,legend_locs=None,figsize=(5,4.5),fontsize=None,file_name=None,show_graphs=True,pcmd='plot',extra_args={}):
    plt.interactive(show_graphs)

    if pcmd == 'bar':
        color_cycle = mpl.rcParams['axes.prop_cycle']
        def pfun(ax,x_data,y_data,**kwargs):
            n_series = len(y_data.T)
            tot_width = np.ptp(x_data)
            width = float(tot_width)/len(x_data)/n_series
            for (i,(cc,y_series)) in enumerate(zip(color_cycle,y_data.T)):
                ax.bar(x_data+(i-float(n_series)/2)*width,y_series,width,**dict(cc,**kwargs))
    else:
        def pfun(ax,x_data,y_data,**kwargs):
            getattr(ax,pcmd)(x_data,y_data,**kwargs)

    n_plots = len(y_vars)
    if y_names is None: y_names = n_plots*[None]
    if y_ranges is None: y_ranges = n_plots*[None]
    if legends is None: legends = n_plots*[None]
    if legend_locs is None: legend_locs = n_plots*[None]

    if type(x_vars) is not list: x_vars = n_plots*[x_vars]
    if type(x_names) is not list: x_names = n_plots*[x_names]
    if type(x_ranges) is not list: x_ranges = n_plots*[x_ranges]
    if type(extra_args) is not list: extra_args = n_plots*[extra_args]

    (rows,cols) = shape
    figx0,figy0 = figsize
    (figx,figy) = (figx0*cols,figy0*rows)
    (fig,axlist) = plt.subplots(rows,cols,figsize=(figx,figy))
    axlist = axlist.flatten()[:n_plots]
    for (xv,yv,xn,yn,xr,yr,lg,ll,ax,ea) in zip(x_vars,y_vars,x_names,y_names,x_ranges,y_ranges,legends,legend_locs,axlist,extra_args):
        x_data = eqvars[xv]
        if type(yv) is not list: yv = [yv]
        y_data = np.array([eqvars[yv1] for yv1 in yv]).T
        pfun(ax,x_data,y_data,**ea)
        ax.locator_params(nbins=7)
        if xn is not None: ax.set_xlabel(xn)
        if xr is not None: ax.set_xlim(xr)
        if yn is not None: ax.set_title(yn)
        if yr is not None: ax.set_ylim(yr)
        if lg is not None: ax.legend(lg,loc=ll if ll is not None else 'best')

    fig.subplots_adjust(bottom=0.15)
    fig.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)

    if not show_graphs:
        plt.close()

def kernel_density_2d(datf,x_var,y_var,x_range=None,y_range=None,graph_squeeze=0.05,log=False,scatter=False,N=64):
    datf_sel = datf[[x_var,y_var]].dropna()
    if x_range is None: x_range = v_range(datf_sel[x_var],graph_squeeze)
    if y_range is None: y_range = v_range(datf_sel[y_var],graph_squeeze)

    gauss_pdf = stats.kde.gaussian_kde(datf_sel.values.T)
    z_mesh = np.mgrid[x_range[0]:x_range[1]:N*1j,y_range[0]:y_range[1]:N*1j].reshape(2,N*N)
    z_vals = gauss_pdf(z_mesh).reshape(N,N)
    if log: z_vals = np.log(z_vals)
    if scatter: plt.scatter(datf_sel[x_var],datf_sel[y_var],alpha=0.5,color='white')
    plt.imshow(z_vals,origin='lower',aspect='auto',extent=[z_mesh[0,:].min(),z_mesh[0,:].max(),z_mesh[1,:].min(),z_mesh[1,:].max()])
    plt.set_cmap(cm.jet)

def datf_eval(datf,formula,use_numpy=True):
    if use_numpy: globs = {'np':np}
    return eval(formula,globs,datf)

def datf_plot(x_vars,y_vars,data,shape=(1,1),figargs={},axargs=None):
    if type(x_vars) is not list: x_vars = [x_vars]
    if type(y_vars) is not list: y_vars = [y_vars]

    (nrows,ncols) = shape
    n_plots = len(x_vars)
    if axargs is None: axargs = n_plots*[{}]

    (fig,axs) = plt.subplots(nrows,ncols,**figargs)
    if type(axs) is not np.ndarray: axs = (axs,)
    axs = axs[:n_plots]

    for (ax,xv,yv,args) in zip(axs,x_vars,y_vars,axargs):
        if type(yv) is not list: yv = [yv]
        x_data = datf_eval(data,xv)
        y_data = np.vstack([datf_eval(data,yvp) for yvp in yv]).T
        ax.plot(x_data,y_data,**args)

# LaTeX Tables
def make_table(format,col_fmts,col_names,col_data,caption='',label='',figure=False):
    col_fmts = ['{:'+cf+'}' for cf in col_fmts]
    col_data = [[cf.format(v) for v in cd] for (cf,cd) in zip(col_fmts,col_data)]
    tcode = ''
    if figure:
        tcode += '\\begin{table}[ht]\n'
        tcode += '\\caption{'+caption+'}\n'
        tcode += '\\label{'+label+'}\n'
    tcode += '\\begin{center}\n'
    tcode += '\\begin{tabular}{'+format+'} \\hline\n'
    tcode += ' & '.join(['\\textbf{'+cn+'}' for cn in col_names])+' \\\\ \\hline\n'
    for row in zip(*col_data): tcode += ' & '.join(row)+' \\\\'+'\n'
    tcode = tcode[:-3]+'\n'
    tcode += '\\end{tabular}\n'
    tcode += '\\end{center}'
    if figure:
        tcode += '\n\\end{table}'
    return tcode

def star_map(pv, star='*'):
    sig = ''
    if pv < 0.1:
        sig += star
    if pv < 0.05:
        sig += star
    if pv < 0.01:
        sig += star
    return sig

def latex_escape(s):
    s1 = s.replace('_', '\\_')
    return s1

# TODO: take extra stats dict, deal with nans
def reg_table_tex(dres, labels={}, note=None, num_fmt='%6.4f', num_func=None, par_func=None, escape=latex_escape):
    if num_func is None: num_func = lambda x: (num_fmt % x) if not np.isnan(x) else ''
    def par_func_def(x):
        ret = num_func(x['param'])
        if not np.isnan(x['pval']):
            ret = '{%s}^{%s}' % (ret, star_map(x['pval']))
        if not np.isnan(x['stder']):
            ret = '$\\begin{array}{c} %s \\\\ (%s) \\end{array}$' % (ret, num_func(x['stder']))
        return ret
    if par_func is None: par_func = par_func_def

    nres = len(dres)
    regs = list(dres)

    info = pd.concat([pd.DataFrame({
        (col, 'param'): res.params,
        (col, 'stder'): np.sqrt(res.cov_params().values.diagonal()),
        (col, 'pval' ): res.pvalues
    }) for col, res in dres.items()], axis=1)
    if len(labels) > 0: info = info.loc[labels].rename(labels)

    tcode = ''
    tcode += '\\begin{tabular}{l%s}\n' % ('c'*nres)
    tcode += '\\toprule\n'
    tcode += '& ' + ' & '.join([escape(s) for s in dres]) + ' \\\\\n'
    tcode += '\\midrule\n'
    tcode += '\\\\\n'
    for (i, v) in info.iterrows():
        vp = v.unstack(level=-1)
        tcode += i +  '& ' + ' & '.join([par_func(x) for i, x in vp[['param', 'stder', 'pval']].loc[regs].iterrows()]) + ' \\\\\n'
        tcode += '\\\\\n'
    tcode += '\\midrule\n'
    tcode += 'N & ' + ' & '.join(['%d' % res.nobs for res in dres.values()]) + ' \\\\\n'
    tcode += '$R^2$ & ' + ' & '.join([num_func(res.rsquared) for res in dres.values()]) + ' \\\\\n'
    tcode += 'Adjusted $R^2$ & ' + ' & '.join([num_func(res.rsquared_adj) for res in dres.values()]) + ' \\\\\n'
    tcode += 'F Statistic & ' + ' & '.join([num_func(res.fvalue) for res in dres.values()]) + ' \\\\\n'
    tcode += '\\bottomrule\n'
    if note is not None: tcode += '\\textit{Note:} & \\multicolumn{%d}{r}{%s}\n' % (nres, escape(note))
    tcode += '\\end{tabular}\n'

    return tcode

latex_template = """\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{booktabs}
\\usepackage[margin=0in]{geometry}
\\begin{document}
\\thispagestyle{empty}
%s
\\end{document}"""
def save_latex(direc, fname, latex, wrap=True, crop=True):
    ttex = latex_template % latex if wrap else latex
    hdir = os.getcwd()
    os.chdir(direc)
    ftex = open('%s.tex' % fname, 'w+')
    ftex.write(ttex)
    ftex.close()
    os.system('latex %s.tex' % fname)
    os.system('dvipdf %s.dvi' % fname)
    if crop: os.system('pdfcrop %s.pdf %s.pdf' % (fname, fname))
    os.chdir(hdir)

def md_escape(s):
    s1 = s.replace('*', '\\*')
    return s1

def reg_table_md(dres, labels={}, order=None, note=None, num_fmt='$%6.4f$', num_func=None, par_func=None, escape=md_escape, fname=None):
    if num_func is None: num_func = lambda x: (num_fmt % x) if not np.isnan(x) else ''
    def par_func_def(x):
        ret = num_func(x['param'])
        if not np.isnan(x['pval']):
            ret += star_map(x['pval'], star='\\*')
        if not np.isnan(x['stder']):
            ret += '<br/>(%s)' % num_func(x['stder'])
        return ret
    if par_func is None: par_func = par_func_def

    nres = len(dres)
    regs = list(dres)

    info = pd.concat([pd.DataFrame({
        (col, 'param'): res.params,
        (col, 'stder'): np.sqrt(res.cov_params().values.diagonal()),
        (col, 'pval' ): res.pvalues
    }) for col, res in dres.items()], axis=1)
    if len(labels) > 0: info = info.loc[labels].rename(labels)

    tcode = ''
    tcode += '| | ' + ' | '.join([escape(s) for s in dres]) + ' |\n'
    tcode += '| - |' + ' - | - '*nres + '|\n'
    for (i, v) in info.iterrows():
        vp = v.unstack(level=-1)
        tcode += '| ' + i +  ' | ' + ' | '.join([par_func(x) for i, x in vp[['param', 'stder', 'pval']].loc[regs].iterrows()]) + ' |\n'
    tcode += '| N | ' + ' | '.join(['$%d$' % res.nobs for res in dres.values()]) + ' |\n'
    tcode += '| $R^2$ | ' + ' | '.join([num_func(res.rsquared) for res in dres.values()]) + ' |\n'
    tcode += '| Adjusted $R^2$ | ' + ' | '.join([num_func(res.rsquared_adj) for res in dres.values()]) + ' |\n'
    tcode += '| F Statistic | ' + ' | '.join([num_func(res.fvalue) for res in dres.values()]) + ' |\n'
    if note is not None: tcode += '*Note:* ' + escape(note)

    if fname is not None:
        with open(fname, 'w+') as fid:
            fid.write(tcode)

    return tcode

def md_table(data, align=None, index=False, fmt='%s'):
    cols = list(data.columns)
    if index:
        cols = [data.index.name or ''] + cols
    if align is None:
        align = 'l'
    if len(align) == 1:
        align = align*len(cols)

    lalign = [' ' if x == 'r' else ':' for x in align]
    ralign = [' ' if x == 'l' else ':' for x in align]

    header = '| ' + ' | '.join([str(x) for x in cols]) + ' |'
    hsep = '|' + '|'.join([la+('-'*max(1,len(x)))+ra for (x, la, ra) in zip(cols, lalign, ralign)]) + '|'
    rows = ['| ' + (str(i)+' | ')*index + ' | '.join([fmt % x for x in row]) + ' |' for (i, row) in data.iterrows()]

    return header + '\n' + hsep + '\n' + '\n'.join(rows)

# sqlite tools

def unfurl(ret,idx=[0]):
    if type(idx) != list: idx = [idx]
    return op.itemgetter(*idx)(zip(*ret))

# time series histogram
def ts_hist(s,agg_type='monthly'):
    plt.style.use('ggplot')

    import calendar
    from datetime import datetime

    # find date range
    values = s.apply(lambda t: (t.year,t.month))
    min_val = min(values)
    max_val = max(values)

    # make bins
    def gen_bins():
        month_incr = lambda y,m: (y,m+1) if m < 12 else (y+1,1)
        val = min_val
        while val != max_val:
            yield val
            val = month_incr(*val)
        yield val
    bins = list(gen_bins())
    nbins = len(bins)

    # determine tick stride
    firstof = lambda m0: min([i for (i,(y,m)) in enumerate(bins) if m == m0])
    if nbins < 12:
        stride = 1
        base = 0
        show_month = True
    elif nbins < 24:
        stride = 2
        base = 0
        show_month = True
    elif nbins < 36:
        stride = 3
        base = min([firstof(1),firstof(4),firstof(7),firstof(10)])
        show_month = True
    elif nbins < 72:
        stride = 6
        base = min([firstof(1),firstof(7)])
        show_month = True
    else:
        stride = 12
        base = firstof(1)
        show_month = False

    # bin it up
    counts = co.OrderedDict([(b,0) for b in bins])
    for val in values:
        counts[val] += 1

    # raw plot
    (fig,ax) = plt.subplots()
    ax.bar(range(nbins),list(counts.values()))
    ticks = range(base,nbins,stride)
    ax.set_xticks(ticks)

    # proper labels
    fmt = '%b %Y' if show_month else '%Y'
    labeler = lambda y,m: datetime(y,m,1).strftime(fmt)
    labels = [labeler(*bins[i]) for i in ticks]
    ax.set_xticklabels(labels,rotation=45)

    # show it
    fig.subplots_adjust(bottom=0.17)

    return (fig,ax)

# time series tools

def shift_column(df, col, per=None, val=None, subset=None, suffix='_p'):
    df1 = df.copy()
    colp = col + '_xxx'
    if val is not None:
        df1[colp] = df[col] + val
    else:
        df1[colp] = df[col].shift(per)
    if subset is None:
        subset = list(df.columns)
    if col not in subset:
        subset += [col]
    df1 = df1.merge(df1[subset], how='left', left_on=colp, right_on=col, suffixes=('', suffix))
    df1 = df1.drop([colp, col+'_p'], axis=1)
    return df1
