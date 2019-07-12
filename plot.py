# plotting tools for lecture

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# defaults
figsize0 = 5, 4

def save_plot(yvars=None, xvar='index', fname=None, data=None, title=None, labels=None, xlabel=None, ylabel=None, legend=True, xlim=None, ylim=None, figsize=figsize0, despine=True, tight=True, facecolor='white'):
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

    (fig,ax) = plt.subplots(figsize=figsize)
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
    if facecolor: fig.set_facecolor(facecolor)
    if despine: sns.despine(ax=ax)

    if fname is None:
        fig.show()
    else:
        save_fig(fig, fname)
        plt.close(fig)

def scatter_label(xvar, yvar, data, labels=True, offset=0.02, ax=None):
    df = data[[xvar, yvar]].dropna()

    if ax is None:
        _, ax = plt.subplots()
    df.plot.scatter(x=xvar, y=yvar, s=0, ax=ax)

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ylim, xlim = ymax - ymin, xmax - xmin

    labvals = df.index if labels in ('index', True) else df[labels]
    for txt in labvals.values:
        point = df[xvar].ix[txt] - offset*xlim, df[yvar].ix[txt] - offset*ylim
        ax.annotate(txt, point)

    return ax

class Diagram():
    def __init__(self, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, xticks=None, yticks=None, xtick_labels=None, ytick_labels=None, **kwargs):
        (self.fig, self.ax) = plt.subplots(**kwargs)

        if title: self.ax.set_title(title)
        if xlabel: self.ax.set_xlabel(xlabel)
        if ylabel: self.ax.set_ylabel(ylabel)
        if xlim: self.ax.set_xlim(xlim)
        if ylim: self.ax.set_ylim(ylim)
        if xticks:
            self.ax.set_xticks(xticks)
            if xtick_labels:
                self.ax.set_xticklabels(xtick_labels)
        else:
            self.ax.set_xticks([])
        if yticks:
            self.ax.set_yticks(yticks)
            if ytick_labels:
                self.ax.set_yticklabels(ytick_labels)
        else:
            self.ax.set_yticks([])

    def line(self, **kwargs):
        return SymbolicLine(self.ax, **kwargs)

    def annotate(self, x, y, text, prop=True, **kwargs):
        if prop:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            xran = xlim[1] - xlim[0]
            yran = ylim[1] - ylim[0]
            x *= xran
            y *= yran
        return self.ax.annotate(text, xy=(x, y), **kwargs)

    def show(self, despine=False):
        sns.despine(fig=self.fig)
        self.fig.show()

    def save(self, fname):
        self.fig.savefig(fname)

fontsize0 = 15
arrowprops0 = {'arrowstyle': '-|>', 'connectionstyle': 'angle3', 'linewidth': 1.2, 'edgecolor': 'black', 'facecolor': 'white'}

class SymbolicLine:
    def __init__(self, ax, eq=None, vert=None):
        self.eq = eq
        self.ax = ax
        self.vert = vert

    def plot(self, xlim=None, ylim=None, N=128, fill=False, fill_color=None, **kwargs):
        if xlim is None:
            xlim = self.ax.get_xlim()
        if ylim is None:
            ylim = self.ax.get_ylim()
        if self.vert is None:
            xvec = np.linspace(xlim[0], xlim[1], N)
            yvec = self.eq(xvec)
        else:
            xvec = self.vert*np.ones(N)
            yvec = np.linspace(ylim[0], ylim[1], N)
        line = self.ax.plot(xvec, yvec, **kwargs)
        if fill:
            self.ax.fill_between(xvec, yvec, color=fill_color)
        return line

    def point(self, x ,**kwargs):
        y = self.eq(x)
        return self.ax.scatter(x, y, **kwargs)

    def annotate(self, text, xy, offset, fontsize=fontsize0, arrowprops=arrowprops0, **kwargs):
        if self.vert is None:
            x = xy
            y = self.eq(xy)
        else:
            x = self.vert
            y = xy
        return self.ax.annotate(text, xy=(x, y), xytext=(x+offset[0], y+offset[1]), fontsize=fontsize, arrowprops=arrowprops, **kwargs)

    def annotate_prop(self, text, xyp, offset, fontsize=fontsize0, arrowprops=arrowprops0, **kwargs):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xran = xlim[1] - xlim[0]
        yran = ylim[1] - ylim[0]
        if self.vert is None:
            xy = xran*xyp
            x = xy
            y = self.eq(x)
        else:
            xy = yran*xyp
            x = self.vert
            y = xy
        return self.ax.annotate(text, xy=(x, y), xytext=(x+xran*offset[0], y+yran*offset[1]), fontsize=fontsize, arrowprops=arrowprops, **kwargs)
