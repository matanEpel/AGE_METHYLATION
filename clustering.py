# TODO: create vector to every cpg island: locations and accordionicity
# TODO: cluster to groups using those vectors and researching on that
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

KNN = 1
AMOUNT_OF_GROUPS = 5
MIN_TO_TYPE = 6

class CLUSTER:
    def __init__(self):
        self.method = KNN

    def cluster_pca(self, vectors, amount_of_groups=AMOUNT_OF_GROUPS):
        # vectors - list of vector
        pca = PCA(n_components=amount_of_groups)
        result = pca.fit_transform(np.array(vectors))
        return np.argmax(np.abs(result),axis=1), (np.max(np.abs(result), axis=1) > MIN_TO_TYPE),np.abs(pca.components_)
