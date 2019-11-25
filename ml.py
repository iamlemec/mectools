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

def vstack2(v, N=1):
    if len(v) == 0:
        return np.array([[]]).reshape((0, N))
    else:
        return np.vstack(v)

# high dimensional fixed effects
# expects strings, expressions, or [N x K] matrices
def sparse_ols(y, x=[], fe=[], data=None, intercept=True):
    def ensure_vector(z):
        r = frame_eval(z, data) if type(z) is str else z
        return r.flatten() if r.ndim > 1 else r
    def gen_names(v):
        return [z if type(z) is str else str(i) for i, z in enumerate(v)]

    # generate output variables
    y_vec = ensure_vector(y)
    N = len(y_vec)

    # generate dense input variables
    if type(x) is np.ndarray:
        x_mat = x
        x_tag = gen_names(range(x.shape[0]))
    else:
        x_mat = vstack2([ensure_vector(z) for z in x], N).T
        x_tag = gen_names(x)

    # add intercept if needed
    if intercept:
        x_mat = np.hstack([np.ones((N, 1)), x_mat])
        x_tag = ['intercept'] + x_tag

    # generate categorical input variables
    if type(fe) is np.ndarray:
        fe_mat = fe
        fe_tag = gen_names(range(fe.shape[0]))
    else:
        fe_mat = vstack2([ensure_vector(z) for z in fe], N).T
        fe_tag = gen_names(fe)

    # sparse one-hot fixed effects
    if fe_mat.size == 0:
        fe_spmat = sp.coo_matrix((N, 0))
        fe_names = []
    else:
        hot = OneHotEncoder(categories='auto', drop='first')
        fe_spmat = hot.fit_transform(fe_mat)
        fe_names = hot.get_feature_names(fe_tag).tolist()

    # compute coefficients
    x_spmat = sp.hstack([x_mat, fe_spmat])
    beta = sp.linalg.spsolve(x_spmat.T*x_spmat, x_spmat.T*y_vec)

    # match names
    names = x_tag + fe_names
    ret = pd.Series(dict(zip(names, beta)))

    return ret

# poisson regression
