from simulations.clustering_iterations import *
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.datasets import load_breast_cancer, make_circles, make_moons, make_blobs, load_wine
from collections import defaultdict
import json
from sklearn import mixture

data = load_wine()['data']
em = mixture.GMM(n_components=3)
em.fit(data)
if hasattr(em, 'labels_'):
    print(0)
else:
    print(em.predict(data))
