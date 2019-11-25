# machine learning and stats stuff

import numpy as np
import scipy.sparse as sp
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from itertools import chain, product

# evaluate in dataframe environemnt
def frame_eval(exp, data, engine='pandas'):
    if engine == 'pandas':
        return data.eval(exp)
    elif engine == 'python':
        return eval(exp, globals(), data)

def vstack2(v, N=1):
    if len(v) == 0:
        return np.array([[]]).reshape((0, N))
    else:
        return np.vstack(v)

def strides(v):
    return np.r_[1, np.cumprod(v[1:])][::-1]

def add_prefix(p, v, sep='_'):
    return [f'{p}{sep}{z}' for z in v]

def chainer(v):
    return list(chain.from_iterable(v))

# need to have all features list -> onehot
# map terms -> feature lists
# terms becomes power sums of features -> then one hot again

# high dimensional fixed effects - expects strings or expressions
def sparse_ols(y, x=[], fe=[], data=None, intercept=True, drop='first'):
    # generate dense input variables
    y_vec = frame_eval(y, data)
    N = len(y_vec)
    x_mat = vstack2([frame_eval(z, data) for z in x], N).T
    fe_mat = vstack2([frame_eval(z, data) for z in fe], N).T

    # add intercept if needed
    if intercept:
        x_mat = np.hstack([np.ones((N, 1)), x_mat])
        x = ['intercept'] + x

    # sparse one-hot fixed effects
    if len(fe) == 0:
        fe_spmat = sp.coo_matrix((N, 0))
        fe_names = []
    else:
        hot = OneHotEncoder(categories='auto', drop=drop)
        fe_spmat = hot.fit_transform(fe_mat)
        fe_names = hot.get_feature_names(fe).tolist()

    # compute coefficients
    x_spmat = sp.hstack([x_mat, fe_spmat])
    beta = sp.linalg.spsolve(x_spmat.T*x_spmat, x_spmat.T*y_vec)

    # match names
    names = x + fe_names
    ret = pd.Series(dict(zip(names, beta)))

    return ret

# high dimensional fixed effects - expects strings or expressions
# can have tuples of strings in fixed effects
def sparse_ols_new(y, x=[], fe=[], data=None, intercept=True, drop='first'):
    # generate output variables
    y_vec = frame_eval(y, data)
    N = len(y_vec)

    # generate input variables
    x_mat = vstack2([frame_eval(z, data) for z in x], N).T
    if intercept:
        x_mat = np.hstack([np.ones((N, 1)), x_mat])
        x = ['intercept'] + x

    # generate map between terms and features
    fe = [[z] if type(z) is str else z for z in fe]
    feat = chainer(fe)
    feat_mat = vstack2([frame_eval(z, data) for z in feat], N).T
    term_tags = [':'.join(term) for term in fe]
    term_map = [np.array([feat.index(z) for z in term]) for term in fe]

    # ordinally encode fixed effects
    enc_ord = OrdinalEncoder(categories='auto')
    feat_ord = enc_ord.fit_transform(feat_mat)
    feat_names = [z.astype(str) for z in enc_ord.categories_]
    feat_size = np.array([len(z) for z in enc_ord.categories_])

    # sparse one-hot fixed effects
    if len(fe) == 0:
        final_spmat = sp.coo_matrix((N, 0))
        final_names = []
    else:
        form_vals = []
        form_names = []
        for term_idx in term_map:
            # generate cross ordinals
            term_sizes = feat_size[term_idx]
            term_stride = strides(term_sizes)
            cross_vals = feat_mat[:,term_idx].dot(term_stride)
            form_vals.append(cross_vals)

            # generate cross names
            term_names = [feat_names[i] for i in term_idx]
            cross_names = [':'.join(x) for x in product(*term_names)]
            form_names.append(cross_names)

        hot = OneHotEncoder(categories='auto', drop=drop)
        final_mat = vstack2(form_vals).T
        final_spmat = hot.fit_transform(final_mat)
        final_names = chainer([f'{t}={z}' for z in n] for t, n in zip(term_tags, form_names))
        print(final_spmat.shape)

    # compute coefficients
    x_spmat = sp.hstack([x_mat, final_spmat])
    beta = sp.linalg.spsolve(x_spmat.T*x_spmat, x_spmat.T*y_vec)

    print(len(beta))
    print(len(final_names))

    # match names
    names = x + final_names
    ret = pd.Series(dict(zip(names, beta)))

    return ret

# poisson regression
