# machine learning and stats stuff

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from itertools import chain, product

# import tensorflow as tf
# import tensorflow.keras as K

# evaluate in dataframe environemnt
def frame_eval(exp, data, engine='pandas'):
    if engine == 'pandas':
        return data.eval(exp)
    elif engine == 'python':
        return eval(exp, globals(), data)

def vstack(v, N=None):
    if len(v) == 0:
        return np.array([[]]).reshape((0, N))
    else:
        return np.vstack(v)

# this assumes row major to align wiht product
def strides(v):
    if len(v) == 1:
        return np.array([1])
    else:
        return np.r_[1, np.cumprod(v[1:])][::-1]

def swizzle(ks, vs):
    return ','.join([f'{k}={v}' for k, v in zip(ks, vs)])

def chainer(v):
    return list(chain.from_iterable(v))

def design_matrix(x, data, N=None):
    return vstack([frame_eval(z, data) for z in x], N).T

def sparse_categorical(terms, data, N=None, drop='first'):
    if len(terms) == 0:
        return sp.coo_matrix((N, 0)), []

    # generate map between terms and features
    terms = [(z,) if type(z) is str else z for z in terms]
    feats = chainer(terms)
    feat_mat = design_matrix(feats, data, N=N)
    term_map = [[feats.index(z) for z in t] for t in terms]

    # ordinally encode fixed effects
    enc_ord = OrdinalEncoder(categories='auto')
    feat_ord = enc_ord.fit_transform(feat_mat).astype(np.int)
    feat_names = [z.astype(str) for z in enc_ord.categories_]
    feat_sizes = [len(z) for z in enc_ord.categories_]

    # generate cross-matrices and cross-names
    form_vals = []
    form_names = []
    for term_idx in term_map:
        # generate cross matrices
        term_sizes = [feat_sizes[i] for i in term_idx]
        term_strides = strides(term_sizes)
        cross_vals = feat_ord[:,term_idx].dot(term_strides)
        form_vals.append(cross_vals)

        # generate cross names
        term_names = [feat_names[i] for i in term_idx]
        cross_names = [x for x in product(*term_names)]
        form_names.append(cross_names)

    # one hot encode all (cross)-terms
    hot = OneHotEncoder(categories='auto', drop=drop)
    final_mat = vstack(form_vals).T
    final_spmat = hot.fit_transform(final_mat)

    # find all cross-term names
    if hot.drop_idx_ is None:
        seen_cats = hot.categories_
    else:
        seen_cats = [np.delete(c, i) for c, i in zip(hot.categories_, hot.drop_idx_)]
    seen_names = [[n[i] for i in c] for c, n in zip(seen_cats, form_names)]
    final_names = chainer([swizzle(t, i) for i in n] for t, n in zip(terms, seen_names))

    return final_spmat, final_names

## high dimensional fixed effects
# x expects strings or expressions
# fe can have strings or tuples of strings
def sparse_ols(y, x=[], fe=[], data=None, intercept=True, drop='first'):
    # generate output variables
    y_vec = frame_eval(y, data)
    N = len(y_vec)

    # find dense variable matrix
    x_mat = design_matrix(x, data, N=N)

    # find sparse categorical matrix
    fe_spmat, fe_names = sparse_categorical(fe, data, N=N, drop=drop)

    # optionally add intercept
    if intercept:
        x_mat = np.hstack([np.ones((N, 1)), x_mat])
        x = ['intercept'] + x

    # compute coefficients
    x_spmat = sp.hstack([x_mat, fe_spmat])
    betas = sp.linalg.spsolve(x_spmat.T*x_spmat, x_spmat.T*y_vec)

    # match names
    names = x + fe_names
    ret = pd.Series(dict(zip(names, betas)))

    return ret

# poisson regression
