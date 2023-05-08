# TODO: create vector to every cpg island: locations and accordionicity
# TODO: cluster to groups using those vectors and researching on that
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

KNN = 1
AMOUNT_OF_GROUPS = 15


class CLUSTER:
    def __init__(self):
        self.method = KNN

    def cluster_pca(self, vectors):
        # vectors - list of vector
        pca = PCA(n_components=AMOUNT_OF_GROUPS)
        result = pca.fit_transform(np.array(vectors))
        return np.argmax(result,axis=1), np.abs(pca.components_)
