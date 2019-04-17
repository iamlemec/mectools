import os
import re
import numpy as np
import pandas as pd
import statsmodels as sm

##
## generic tables
##

def to_latex(data, align=None, index=False, fmt='%s'):
    data = data.copy()
    cols = list(data.columns)

    def to_string(x):
        if x.dtype in (np.float32, np.float64):
            return x.apply(lambda x: fmt % x)
        else:
            return x.apply(str)

    data = data.apply(to_string)
    data.index = to_string(pd.Series(data.index))

    if index:
        cols = [data.index.name or ''] + cols
    if align is None:
        align = 'l'
    if len(align) == 1:
        align = align*len(cols)

    header = ' & '.join([f'\\textbf{{{c}}}' for c in cols])
    rows = [(i+' & ')*index + ' & '.join(row) for i, row in data.iterrows()]

    tcode = ''
    tcode += '\\begin{tabular}{' + align + '} \\hline\n'
    tcode += header + ' \\\\ \\hline\n'
    tcode += ' \\\\\n'.join(rows) + '\n'
    tcode += '\\end{tabular}'

    return tcode

def to_markdown(data, align=None, index=False, fmt='%s'):
    data = data.copy()
    cols = list(data.columns)

    def to_string(x):
        if x.dtype in (np.float32, np.float64):
            return x.apply(lambda x: fmt % x)
        else:
            return x.apply(str)

    data = data.apply(to_string)
    data.index = to_string(pd.Series(data.index))

    if index:
        cols = [data.index.name or '-'] + cols
    if align is None:
        align = 'l'
    if len(align) == 1:
        align = align*len(cols)

    lalign = [' ' if x == 'r' else ':' for x in align]
    ralign = [' ' if x == 'l' else ':' for x in align]

    header = '| ' + ' | '.join([str(x) for x in cols]) + ' |'
    hsep = '|' + '|'.join([la+('-'*max(1,len(x)))+ra for x, la, ra in zip(cols, lalign, ralign)]) + '|'
    rows = ['| ' + (i+' | ')*index + ' | '.join(row) + ' |' for i, row in data.iterrows()]

    return header + '\n' + hsep + '\n' + '\n'.join(rows)

# for obscure but not unheard of use cases
def from_markdown(md):
    split_row = lambda row: [s.strip() for s in row.split('|')[1:-1]]
    lines = md.strip().split('\n')
    head, dat = lines[0], lines[2:]
    cols = split_row(head)
    vals = [split_row(row) for row in dat]
    frame = pd.DataFrame(vals, columns=cols)
    for c in frame:
        try:
            frame[c] = pd.to_numeric(frame[c])
        except:
            pass
    return frame

##
## star tables
##

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
    s = re.sub(r'([_&])', r'\\\1', s)
    return s

stats0 = {
    'N': 'nobs',
    '$R^2$': 'rsquared',
    'Adjusted $R^2$': 'rsquared_adj',
    'F Statistic': 'fvalue'
}

def reg_dict(res):
    return pd.concat({
        'param': res.params,
        'stderr': pd.Series(np.sqrt(res.cov_params().values.diagonal()), index=res.model.exog_names),
        'pvalue': res.pvalues,
    }, axis=1)

def reg_stats(res, stats={}):
    return pd.Series({lab: getattr(res, att, np.nan) for lab, att in stats.items()})

# TODO: take extra stats dict, deal with nans
def regtab_latex(info, labels=None, columns=None, note=None, num_fmt='%6.4f', num_func=None, par_func=None, escape=latex_escape, stats=stats0, save=None):
    def num_func_def(x):
        if np.isnan(x):
            return ''
        elif type(x) in (np.int, np.int64):
            return '%d' % x
        elif type(x) in (np.float, np.float64):
            return num_fmt % x
        else:
            return str(x)
    if num_func is None:
        num_func = num_func_def

    def par_func_def(x):
        ret = num_func(x['param'])
        if not np.isnan(x['pvalue']):
            ret = '{%s}^{%s}' % (ret, star_map(x['pvalue']))
        if not np.isnan(x['stderr']):
            ret = '$\\begin{array}{c} %s \\\\ (%s) \\end{array}$' % (ret, num_func(x['stderr']))
        return ret
    if par_func is None:
        par_func = par_func_def

    # see if it's a dict of regression results and if so turn it into a table
    # with (reg, stat) columns and (exog_name) rows. otherwise, should aleady
    # be one of these.
    if type(info) is dict:
        stats = pd.concat({col: reg_stats(res, stats) for col, res in info.items()}, axis=1)
        info = pd.concat({col: reg_dict(res) for col, res in info.items()}, axis=1)

    # handle column name and order
    if columns is not None:
        corder = list(columns.values())
        info = info[list(columns)]
        if type(columns) is dict:
            info = info.rename(columns, axis=1)
    else:
        corder = list(info.columns.levels[0])
    ncol = len(corder)

    # handle row name and order
    if labels is not None:
        lorder = list(labels.values())
        info = info.loc[list(labels)]
        if type(labels) is dict:
            info = info.rename(labels, axis=0)
    else:
        lorder = list(info.index)
    nrow = len(lorder)

    tcode = ''
    tcode += '\\begin{tabular}{l%s}\n' % ('c'*ncol)
    tcode += '\\toprule\n'
    tcode += '& ' + ' & '.join([escape(s) for s in corder]) + ' \\\\\n'
    tcode += '\\midrule\n'
    tcode += '\\\\\n'
    for i, v in info.iterrows():
        vp = v.unstack(level=-1)
        tcode += escape(i) +  ' & ' + ' & '.join([par_func(x) for j, x in vp[['param', 'stderr', 'pvalue']].loc[corder].iterrows()]) + ' \\\\\n'
        tcode += '\\\\\n'
    tcode += '\\midrule\n'
    if stats is not None:
        for i, v in stats.iterrows():
            tcode += escape(i) + ' & ' + ' & '.join([num_func(x) for j, x in v.iteritems()]) + ' \\\\\n'
    tcode += '\\bottomrule\n'
    if note is not None:
        tcode += '\\textit{Note:} & \\multicolumn{%d}{r}{%s}\n' % (ncol, escape(note))
    tcode += '\\end{tabular}'

    if save is None:
        return tcode
    else:
        with open(save, 'w+') as fout:
            fout.write(tcode)

latex_template = """\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{booktabs}
\\usepackage[margin=1in]{geometry}
\\begin{document}
\\thispagestyle{empty}

%s

\\end{document}"""

def save_latex(latex, fname, direc=None, wrap=True, crop=False):
    if direc is None:
        direc, fname = os.path.split(fname)
    if direc != '':
        hdir = os.getcwd()
        os.chdir(direc)
    else:
        hdir = None

    with open('%s.tex' % fname, 'w+') as fid:
        fid.write(latex_template % latex if wrap else latex)

    os.system('pdflatex %s.tex' % fname)

    if crop:
        os.system('pdfcrop %s.pdf %s.pdf' % (fname, fname))

    if hdir is not None:
        os.chdir(hdir)

def markdown_escape(s):
    s1 = s.replace('*', '\\*')
    return s1

def regtab_markdown(info, labels={}, order=None, note=None, num_fmt='%6.4f', num_func=None, par_func=None, escape=markdown_escape, stats=stats0, fname=None):
    def num_func_def(x):
        if np.isnan(x):
            return ''
        elif type(x) in (np.int, np.int64):
            return '$%d$' % x
        elif type(x) in (np.float, np.float64):
            return '$' + (num_fmt % x) + '$'
        else:
            return str(x)
    if num_func is None:
        num_func = num_func_def

    def par_func_def(x):
        ret = num_func(x['param'])
        if not np.isnan(x['pval']):
            ret += star_map(x['pval'], star='\\*')
        if not np.isnan(x['stder']):
            ret += '<br/>(%s)' % num_func(x['stder'])
        return ret
    if par_func is None:
        par_func = par_func_def

    nres = len(info)
    regs = list(info)

    data = pd.concat([pd.DataFrame({
        (col, 'param'): res.params,
        (col, 'stder'): np.sqrt(res.cov_params().values.diagonal()),
        (col, 'pval' ): res.pvalues
    }) for col, res in info.items()], axis=1)
    if len(labels) > 0: data = data.loc[labels].rename(labels)

    tcode = ''
    tcode += '| - | ' + ' | '.join([escape(s) for s in info]) + ' |\n'
    tcode += '| - |' + ' - |'*nres + '\n'
    for i, v in data.iterrows():
        vp = v.unstack(level=-1)
        tcode += '| ' + i +  ' | ' + ' | '.join([par_func(x) for i, x in vp[['param', 'stder', 'pval']].loc[regs].iterrows()]) + ' |\n'
    for lab, att in stats.items():
        tcode += '| ' + lab + ' | ' + ' | '.join([num_func(getattr(res, att, np.nan)) for res in info.values()]) + ' |\n'
    if note is not None:
        tcode += '*Note:* ' + escape(note)

    if fname is not None:
        with open(fname, 'w+') as fid:
            fid.write(tcode)

    return tcode

def graph_latex(img, args=''):
    if args is not None:
        args = f'[{args}]'
    return '\\includegraphics%s{%s}' % (args, img)

float_template = """\\begin{%(env)s}%(pos)s
\\centering
%(label)s%(caption)s%(text)s
\\end{%(env)s}"""

def float_latex(env, data=None, text=None, label=None, caption=None, pos='h'):
    if label is not None:
        label = '\\label{%s}\n' % label
    else:
        label = ''
    if caption is not None:
        caption = '\\caption{%s}\n' % caption
    else:
        caption = ''
    if pos in [None, '']:
        pos = ''
    else:
        pos = f'[{pos}]'

    if text is None:
        if env == 'table':
            text = regtab_latex(**data)
        elif env == 'figure':
            text = graph_latex(**data)

    return float_template % {
        'env': env,
        'pos': pos,
        'label': label,
        'caption': caption,
        'text': text
    }

def report_latex(info, save=None):
    if type(info) is dict:
        info = [{'label': k, **v} for k, v in info.items()]
    body = '\n\n'.join([float_latex(**fig) for fig in info])
    if save:
        save_latex(body, save)
    else:
        return body
