# machine learning and stats stuff

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from itertools import chain, product

##
## tools
##

def frame_eval(exp, data, engine='pandas'):
    if engine == 'pandas':
        return data.eval(exp).values
    elif engine == 'python':
        return eval(exp, globals(), data).values

def vstack(v, N=None):
    if len(v) == 0:
        return np.array([[]]).reshape((0, N))
    else:
        return np.vstack(v)

# this assumes row major to align with product
def strides(v):
    if len(v) == 1:
        return np.array([1])
    else:
        return np.r_[1, np.cumprod(v[1:])][::-1]

def swizzle(ks, vs):
    return ','.join([f'{k}={v}' for k, v in zip(ks, vs)])

def chainer(v):
    return list(chain.from_iterable(v))

##
## design matrices
##

def frame_matrix(x, data, N=None):
    return vstack([frame_eval(z, data) for z in x], N).T

def sparse_categorical(terms, data, N=None, drop='first'):
    if len(terms) == 0:
        return sp.csr_matrix((N, 0)), []

    # generate map between terms and features
    terms = [(z,) if type(z) is str else z for z in terms]
    feats = chainer(terms)
    feat_mat = frame_matrix(feats, data, N=N)
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

def design_matrix(x=[], fe=[], data=None, intercept=True, drop='first', N=None):
    # construct individual matrices
    if len(x) > 0:
        x_mat, x_names = frame_matrix(x, data, N=N), x.copy()
    else:
        x_mat, x_names = None, []
    if len(fe) > 0:
        fe_mat, fe_names = sparse_categorical(fe, data, drop=drop, N=N)
    else:
        fe_mat, fe_names = None, []

    # try to infer N
    if N is None:
        if x_mat is not None:
            N, _ = x_mat.shape
        elif fe_mat is not None:
            N, _ = fe_mat.shape

    # we should know N by now
    if N is None:
        raise(Exception('Must specify N if no data'))

    # optionally add intercept
    if intercept:
        inter = np.ones((N, 1))
        x_mat = np.hstack([inter, x_mat]) if x_mat is not None else inter
        x_names = ['intercept'] + x_names

    # handle various cases
    names = x_names + fe_names
    if x_mat is not None and fe_mat is not None:
        mat = sp.hstack([x_mat, fe_mat], format='csr')
    elif x_mat is not None and fe_mat is None:
        mat = x_mat
    elif x_mat is None and fe_spmat is not None:
        mat = fe_mat
    else:
        mat = np.empty((N, 0))

    # return everything
    return mat, names

def design_matrices(y, x=[], fe=[], data=None, intercept=True, drop='first'):
    y_vec = frame_eval(y, data)
    N = len(y_vec)
    x_mat, x_names = design_matrix(x, fe, data, N=N, intercept=intercept, drop=drop)
    return y_vec, x_mat, x_names

##
## regressions
##

## high dimensional fixed effects
# x expects strings or expressions
# fe can have strings or tuples of strings
def ols(y, x=[], fe=[], data=None, intercept=True, drop='first'):
    y_vec, x_mat, x_names = design_matrices(y, x, fe, data, intercept=intercept, drop=drop)
    if sp.issparse(x_mat):
        betas = sp.linalg.spsolve(x_mat.T*x_mat, x_mat.T*y_vec)
    else:
        betas = np.linalg.solve(np.dot(x_mat.T, x_mat), np.dot(x_mat.T, y_vec))
    ret = pd.Series(dict(zip(x_names, betas)))
    return ret

# feed in sparse matrix in dense batches
class SparseDataset(keras.utils.Sequence):
    def __init__(self, y_vec, x_mat, batch_size):
        self.x_mat = x_mat
        self.y_vec = y_vec
        self.batch_size = batch_size
        self.num_batch = int(np.floor(len(y_vec)/float(batch_size)))

    def __len__(self):
        return self.num_batch

    def __getitem__(self, idx):
        base = idx*self.batch_size
        top = base + self.batch_size

        x_bat = self.x_mat[base:top,:].todense()
        y_bat = self.y_vec[base:top]

        return x_bat, y_bat

# poisson regression using keras
def poisson(y, x=[], fe=[], data=None, intercept=True, drop='first',
            batch_size=4092, epochs=3, learning_rate=0.5):
    # construct design matrices
    y_vec, x_mat, names = design_matrices(y, x, fe, data, intercept=intercept, drop=drop)
    _, K = x_mat.shape

    # construct network
    inputs = layers.Input((K,), name='x')
    lpred = layers.Dense(1, use_bias=False)(inputs)
    pred = keras.layers.Lambda(tf.exp)(lpred)
    model = keras.Model(inputs=inputs, outputs=pred)

    # run estimation
    optim = keras.optimizers.Adagrad(learning_rate=learning_rate)
    model.compile(loss='poisson', optimizer=optim, metrics=['accuracy'])
    if sp.issparse(x_mat):
        dataset = SparseDataset(y_vec, x_mat, batch_size)
        model.fit_generator(dataset, epochs=epochs)
    else:
        model.fit(x_mat, y_vec, epochs=epochs, batch_size=batch_size)

    # extract coefficients
    betas = model.weights[0].numpy().flatten()
    ret = pd.Series(dict(zip(names, betas)))
    return ret
