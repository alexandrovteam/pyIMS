import numpy as np
def random_projection(X, n=100):
    """
    :param X: data matrix 
            n: number of components to return
    :return: reduced data
    """
    import numpy as np
    return np.dot(np.random.randn(n, X.shape[0]), X).T

def tsne(X, n=3):
    """
    :param X: data matrix 
    :return: reduced data
    """
    from sklearn.manifold import TSNE
    model = TSNE(n_components=n, random_state=0)
    np.set_printoptions(suppress=True)
    return model.fit_transform(X)
