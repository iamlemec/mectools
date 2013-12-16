# misc data tools

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.cm as cm
import patsy
import vincent

# statistics

def noinf(s):
  return s.replace([-np.inf,np.inf],np.nan)

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
  print header_fmt.format('',*df.columns)
  for (i,vs) in df.iterrows(): print row_fmt.format(str(i),*vs.values)

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
  print svar.describe()
  svar.hist()

def corr_info(datf,x_var,y_var,w_var=None,c_var='index',x_range=None,y_range=None,x_name=None,y_name=None,title='',reg_type=None,size_scale=1.0,winsor=None,graph_squeeze=0.05,alpha=0.8,color_skew=0.5,fontsize=None):
  all_vars = [x_var,y_var]
  if w_var: all_vars += [w_var]
  if c_var and not c_var == 'index': all_vars += [c_var]

  datf_sel = datf[all_vars].dropna()
  if winsor is not None:
    datf_sel[[x_var,y_var]] = winsorize(datf_sel[[x_var,y_var]],level=winsor)

  if reg_type is None:
    if w_var is None:
      reg_type = 'OLS'
    else:
      reg_type = 'WLS'

  if x_name is None: x_name = x_var
  if y_name is None: y_name = y_var

  if x_range is None: x_range = v_range(datf_sel[x_var],graph_squeeze)
  if y_range is None: y_range = v_range(datf_sel[y_var],graph_squeeze)

  if reg_type == 'WLS':
    mod = sm.WLS(datf_sel[y_var],sm.add_constant(datf_sel[x_var]),weights=datf_sel[w_var])
  else: # OLS + others
    reg_unit = getattr(sm,reg_type)
    mod = reg_unit(datf_sel[y_var],sm.add_constant(datf_sel[x_var]))
  res = mod.fit()

  x_vals = np.linspace(x_range[0],x_range[1],128)
  y_vals = res.predict(sm.add_constant(x_vals))

  (corr,corr_pval) = corr_robust(datf_sel,x_var,y_var,wcol=w_var)

  str_width = max(11,len(x_var))
  fmt_0 = '{:'+str(str_width)+'s} = {: f}'
  fmt_1 = '{:'+str(str_width)+'s} = {: f} ({:f})'
  print fmt_0.format('constant',res.params[1])
  print fmt_1.format(x_var,res.params[0],res.pvalues[0])
  print fmt_1.format('correlation',corr,corr_pval)
  #print fmt_0.format('R-squared',res.rsquared)

  if w_var:
    wgt_norm = datf_sel[w_var]
    wgt_norm /= np.mean(wgt_norm)
  else:
    wgt_norm = np.ones_like(datf_sel[x_var])

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

  (fig,ax) = plt.subplots()
  ax.scatter(datf_sel[x_var],datf_sel[y_var],s=20.0*size_scale*wgt_norm,color=color_vals,alpha=alpha)
  ax.plot(x_vals,y_vals,color='r',linewidth=1.0,alpha=0.7)
  ax.set_xlim(x_range)
  ax.set_ylim(y_range)
  ax.set_xlabel(x_name,fontsize=fontsize)
  ax.set_ylabel(y_name,fontsize=fontsize)
  ax.set_title(title,fontsize=fontsize)

  return res

def grid_plots(eqvars,x_vars,y_vars,shape,x_names=None,y_names=None,x_ranges=None,y_ranges=None,legends=None,legend_locs=None,figsize=(4,3.5),file_name=None,show_graphs=True,pcmd='plot',extra_args={}):
  plt.interactive(show_graphs)

  if pcmd == 'bar':
    color_cycle = mpl.rcParams['axes.color_cycle']
    def pfun(ax,x_data,y_data,**kwargs):
      n_series = len(y_data.T)
      tot_width = np.ptp(x_data)
      width = float(tot_width)/len(x_data)/n_series
      for (i,y_series) in enumerate(y_data.T):
        ax.bar(x_data+(i-float(n_series)/2)*width,y_series,width,**dict({'color':color_cycle[i]},**kwargs))
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
