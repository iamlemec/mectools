# machine learning and stats stuff

import numpy as np
import scipy.sparse as sp
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
from sklearn.preprocessing import OneHotEncoder

# evaluate in dataframe environemnt
def frame_eval(exp, data, engine='pandas'):
    if engine == 'pandas':
        return data.eval(exp)
    elif engine == 'python':
        return eval(exp, globals(), data)

# high dimensional fixed effects
# expects strings, expressions, or [N x K] matrices
def sparse_ols(y, x, fe=[], data=None):
    def ensure_vector(z):
        r = frame_eval(z, data) if type(z) is str else z
        return r.flatten() if r.ndim > 1 else r
    def gen_names(v):
        return [z if type(z) is str else str(i) for i, z in enumerate(v)]

    # generate intermediate matrices
    y_vec = ensure_vector(y)
    if type(x) is np.ndarray:
        x_mat = x
        x_tag = gen_names(range(x.shape[0]))
    else:
        x_mat = np.vstack([ensure_vector(z) for z in x]).T
        x_tag = gen_names(x)
    if type(fe) is np.ndarray:
        fe_mat = fe
        fe_tag = gen_names(range(fe.shape[0]))
    else:
        fe_mat = np.vstack([ensure_vector(z) for z in fe]).T
        fe_tag = gen_names(fe)

    # check shapes
    assert(y_vec.shape[0] == x_mat.shape[0] == fe_mat.shape[0])
    N = len(y_vec)

    # sparse one-hot fixed effects
    hot = OneHotEncoder(categories='auto')
    fe_spmat = hot.fit_transform(fe_mat)
    fe_names = hot.get_feature_names(fe_tag)

    # compute coefficients
    X_mat = sp.hstack([x_mat, fe_spmat])
    beta = sp.linalg.spsolve(X_mat.T*X_mat, X_mat.T*y_vec)

    # match names
    names = x_tag + list(fe_names)
    ret = pd.Series(dict(zip(names, beta)))

    return ret

# poisson regression
