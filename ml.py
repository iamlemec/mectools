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

def vstack(v, N=1):
    if len(v) == 0:
        return np.array([[]]).reshape((0, N))
    else:
        return np.vstack(v)

def strides(v):
    if len(v) == 1:
        return np.array([1])
    else:
        return np.r_[1, np.cumprod(v[1:])][::-1]

def prefix(p, v):
    return [f'{p}{z}' for z in v]

def chainer(v):
    return list(chain.from_iterable(v))

# need to have all features list -> onehot
# map terms -> feature lists
# terms becomes power sums of features -> then one hot again

# high dimensional fixed effects - expects strings or expressions
def sparse_ols_simple(y, x=[], fe=[], data=None, intercept=True, drop='first'):
    # generate dense input variables
    y_vec = frame_eval(y, data)
    N = len(y_vec)
    x_mat = vstack([frame_eval(z, data) for z in x], N).T
    fe_mat = vstack([frame_eval(z, data) for z in fe], N).T

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
def sparse_ols_cross(y, x=[], fe=[], data=None, intercept=True, drop='first'):
    # generate output variables
    y_vec = frame_eval(y, data)
    N = len(y_vec)

    # generate input variables
    x_mat = vstack([frame_eval(z, data) for z in x], N).T
    if intercept:
        x_mat = np.hstack([np.ones((N, 1)), x_mat])
        x = ['intercept'] + x

    # generate map between terms and features
    fe = [[z] if type(z) is str else z for z in fe]
    feat = chainer(fe)
    feat_mat = vstack([frame_eval(z, data) for z in feat], N).T
    term_tag = [':'.join(term) for term in fe]
    term_map = [[feat.index(z) for z in term] for term in fe]

    # ordinally encode fixed effects
    enc_ord = OrdinalEncoder(categories='auto')
    feat_ord = enc_ord.fit_transform(feat_mat).astype(np.int)
    feat_name = [z.astype(str) for z in enc_ord.categories_]
    feat_size = [len(z) for z in enc_ord.categories_]

    if len(fe) == 0:
        final_spmat = sp.coo_matrix((N, 0))
        final_name = []
    else:
        form_val = []
        form_name = []
        for term_idx in term_map:
            # generate cross ordinals
            term_size = [feat_size[i] for i in term_idx]
            term_stride = strides(term_size)
            cross_val = feat_ord[:,term_idx].dot(term_stride)
            form_val.append(cross_val)

            # generate cross names
            term_name = [feat_name[i] for i in term_idx]
            cross_name = np.array([':'.join(x) for x in product(*term_name)])
            form_name.append(cross_name)

        # one hot encode all (cross)-terms
        hot = OneHotEncoder(categories='auto', drop=drop)
        final_mat = vstack(form_val).T
        final_spmat = hot.fit_transform(final_mat)
        if hot.drop_idx_ is None:
            seen_cats = hot.categories_
        else:
            seen_cats = [np.delete(c, i) for c, i in zip(hot.categories_, hot.drop_idx_)]
        seen_name = [n[c] for c, n in zip(seen_cats, form_name)]
        final_name = chainer(prefix(f'{t}=', n) for t, n in zip(term_tag, seen_name))


    # compute coefficients
    x_spmat = sp.hstack([x_mat, final_spmat])
    beta = sp.linalg.spsolve(x_spmat.T*x_spmat, x_spmat.T*y_vec)

    # match names
    name = x + final_name
    ret = pd.Series(dict(zip(name, beta)))

    return ret

# poisson regression
