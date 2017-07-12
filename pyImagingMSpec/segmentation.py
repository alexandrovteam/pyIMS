import numpy as np
def dbscan(X, eps=0.3, return_coefficients=False):
    """
    
    :param X: data matrix 
    :return: Y: segmentation map
             c: coefficients 
    """
    from sklearn.cluster import DBSCAN
    from .dimensionalityReduction import random_projection, tsne
    Xr = tsne(random_projection(X))
    print Xr.shape, eps
    db = DBSCAN(eps=eps, min_samples=10).fit(Xr)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    return db.labels_

