import sqlite3
import scipy.sparse as sp
import pandas as pd
from sklearn.feature_extraction.text as fe

##
## sparse word frequency matrices
##

def weight_counts(counts, weights=None, norm='l2'):
    ndoc, ntok = counts.shape

    # compute tfidf weights
    if weights is None:
        df = (counts>0).sum(axis=0).getA1()
        weights = np.log((ndoc+1)/(df+1)) + 1

    # apply weights
    counts = counts*sp.spdiags(weights, 0, ntok, ntok)

    # apply l2 normalization
    if norm == 'l2':
        l2sum = counts.multiply(counts).sum(axis=1).getA1()
        counts = sp.spdiags(1.0/np.sqrt(l2sum), 0, ndoc, ndoc)*counts

    return counts
