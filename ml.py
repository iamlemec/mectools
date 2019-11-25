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

# need to have all features list -> onehot
# map terms -> feature lists
# terms becomes power sums of features -> then one hot again

# high dimensional fixed effects
# expects strings, expressions, or [N x K] matrices
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

# poisson regression
