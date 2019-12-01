# machine learning and stats stuff

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
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

def negate(f):
    return lambda *x, **kwargs: -f(*x, **kwargs)

##
## tensorflow tools
##

# make a sparse tensor
def sparse_tensor(inp, dtype=np.float32):
    mat = inp.tocoo().astype(dtype)
    idx = list(zip(mat.row, mat.col))
    ten = tf.SparseTensor(idx, mat.data, mat.shape)
    return tf.sparse.reorder(ten)

# dense layer taking sparse matrix as input
class SparseLayer(layers.Layer):
    def __init__(self, vocabulary_size, num_units, activation=None, use_bias=True, **kwargs):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.num_units = num_units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[self.vocabulary_size, self.num_units]
        )
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.num_units])

    def call(self, inputs, **kwargs):
        is_sparse = isinstance(inputs, tf.SparseTensor)
        matmul = tf.sparse.sparse_dense_matmul if is_sparse else tf.matmul
        inters = matmul(inputs, self.kernel)
        outputs = tf.add(inters, self.bias) if self.use_bias else inters
        return self.activation(outputs)

    def compute_output_shape(self, input_shape):
        input_shape = input_shape.get_shape().as_list()
        return input_shape[0], self.num_units

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

def design_matrix(x=[], fe=[], data=None, intercept=True, drop='first', separate=False, N=None):
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

    # if sparse/dense separate we're done
    if separate:
        return x_mat, fe_mat, x_names, fe_names

    # handle various cases
    if x_mat is not None and fe_mat is not None:
        mat = sp.hstack([x_mat, fe_mat], format='csr')
    elif x_mat is not None and fe_mat is None:
        mat = x_mat
    elif x_mat is None and fe_spmat is not None:
        mat = fe_mat
    else:
        mat = np.empty((N, 0))

    # return everything
    names = x_names + fe_names
    return mat, names

def design_matrices(y, x=[], fe=[], data=None, intercept=True, drop='first', separate=False):
    y_vec = frame_eval(y, data)
    N = len(y_vec)
    x_ret = design_matrix(x, fe, data, N=N, intercept=intercept, drop=drop, separate=separate)
    return (y_vec,) + x_ret

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

# glm regression using keras
def glm(y, x=[], fe=[], data=None, intercept=True, drop='first',
        output='params', link=None, loss='mse', like=None, batch_size=4092,
        epochs=3, learning_rate=0.5, metrics=['accuracy']):
    if type(link) is str:
        link = getattr(K, link)
    if type(loss) is str:
        loss = getattr(keras.losses, loss)

    if like is None and loss is not None:
        like = negate(loss)
    if loss is None and like is not None:
        loss = negate(like)

    # construct design matrices
    y_vec, x_mat, fe_mat, x_names, fe_names = design_matrices(
        y, x, fe, data, intercept=intercept, drop=drop, separate=True
    )

    # collect model components
    x_data = [] # actual data
    inputs = [] # input placeholders
    activ = [] # activation layers
    outputs = [] # activation tensors
    names = [] # coefficient names

    # check dense factors
    if x_mat is not None:
        _, Kd = x_mat.shape
        inputs_dense = layers.Input((Kd,))
        linear_dense = layers.Dense(1, use_bias=False)
        pred_dense = linear_dense(inputs_dense)

        x_data.append(x_mat)
        inputs.append(inputs_dense)
        activ.append(linear_dense)
        outputs.append(pred_dense)
        names.append(x_names)

    # check sparse factors
    if fe_mat is not None:
        _, Ks = fe_mat.shape
        fe_ten = sparse_tensor(fe_mat)
        inputs_sparse = layers.Input((Ks,), sparse=True)
        linear_sparse = SparseLayer(Ks, 1, use_bias=False)
        pred_sparse = linear_sparse(inputs_sparse)

        x_data.append(fe_ten)
        inputs.append(inputs_sparse)
        activ.append(linear_sparse)
        outputs.append(pred_sparse)
        names.append(fe_names)

    # construct network
    if len(outputs) > 1:
        pred = layers.Add()(outputs)
    else:
        pred, = outputs
    if link is not None:
        pred = keras.layers.Lambda(link)(pred)
    model = keras.Model(inputs=inputs, outputs=pred)

    # run estimation
    optim = keras.optimizers.Adagrad(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    model.fit(x_data, y_vec, epochs=epochs, batch_size=batch_size)

    # construct params
    names = sum(names, [])
    betas = tf.concat([tf.reshape(act.weights[0], (-1,)) for act in activ], 0)
    ret = pd.Series(dict(zip(names, betas.numpy())))

    # return
    if output == 'params':
        return ret
    elif output == 'model':
        return model, ret

# standard poisson regression
def poisson(y, x=[], fe=[], data=None, **kwargs):
    return glm(y, x=x, fe=fe, data=data, link='exp', loss='poisson', **kwargs)

# try negative binomial next (custom loss)
