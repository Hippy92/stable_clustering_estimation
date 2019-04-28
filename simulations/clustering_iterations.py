import numpy as np
from scipy.stats import norm
from operator import mul
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, load_iris

# Generate the data to test
# linalg = np.linalg

# zeroes = np.zeros([200, 2])
# N = 100
# mean = [1, 1]
# cov = [[0.3, 0.2], [0.2, 0.5]]
# zeroes[0:100, :] = np.random.multivariate_normal(mean, cov, N)
#
#
# mean2 = [5, 5]
# cov2 = [[0.3, 0.2], [0.2, 0.1]]
# zeroes[100:200, :] = np.random.multivariate_normal(mean2, cov2, N)

'''Centric circules'''
# theta = np.random.uniform(0, 2*np.pi, 100)
# r = np.random.uniform(2., 3.5, 100)
# x = r*np.cos(theta)
# y = r*np.sin(theta)
#
# r2 = np.random.uniform(10., 7.5, 100)
# x2 = r2*np.cos(theta)
# y2 = r2*np.sin(theta)

# plt.scatter(zeroes[0:100, 0], zeroes[0:100, 1], c='yellow')
# plt.scatter(zeroes[100:200, 0], zeroes[100:200, 1], c='green')

# plt.scatter(x, y, c='green')
# plt.scatter(x2, y2, c='blue')
# plt.show()

# kmeans = KMeans(n_clusters=4, random_state=0).fit(zeroes)
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)

# data = load_iris()
# print(data['target'])
# normal_data = data['data'] / data['data'].max(axis=0)
#
# kmeans = KMeans(n_clusters=3, random_state=0).fit(normal_data)
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)


def data_noises_generation(data: np.array, bounds: float):
    """
    Noises are generated from uniform distribution
    Symetric parts of avg_vals by column
    """
    avg = data.mean(axis=0)
    zeros = np.zeros(data.shape)
    for idx, mean in enumerate(avg):
        zeros[:, idx] = np.random.uniform(-mean*bounds, mean*bounds, (data.shape[0],))
    return zeros


def generate_noised_sample(data: np.array, bounds: float):
    """
    Noised data generation
    """
    return data + data_noises_generation(data, bounds)


def convert_clusters_to_matrix(clusters_list: list):
    """
    Calculate the cluster similarity matrix
    """
    shape = (len(clusters_list), len(clusters_list))
    matrix_c = np.zeros(shape)

    for i, v_i in enumerate(clusters_list):
        for j, v_j in enumerate(clusters_list):
            matrix_c[i, j] = 1 if v_i == v_j else 0

    return matrix_c


def clustering_simulation(data: np.array, clustering_algorithm):
    """
    Calculate clusters, return cluster similarity matrix C
    """

    clustering_algorithm.fit(data)
    if hasattr(clustering_algorithm, 'labels_'):
        return convert_clusters_to_matrix(clustering_algorithm.labels_)
    else:
        return convert_clusters_to_matrix(clustering_algorithm.predict(data))


def scalar_mult(a, b):
    return np.multiply(a, b).sum()


# ===================== Distance functions =====================
def cosine_distance(a, b):
    return 1 - scalar_mult(a, b) / np.sqrt(scalar_mult(a, a) * scalar_mult(b, b))


def jaccar_distance(a, b):
    return 1 - scalar_mult(a, b) / (scalar_mult(a, a) + scalar_mult(b, b) - scalar_mult(a, b))


def m_distance(a, b):
    return scalar_mult(a - b,  a - b) / (a.shape[0])**2


def lb_distance(a, b):
    return np.all(a == b)


def similarities_calc(noised_matrices_c, original_c, distance):
    """
    Similarity vector
    """

    return [distance(original_c, c) for c in noised_matrices_c]


def bootstrap(n_simulations, t_vect):
    """
    Generate the sample of cluster avg similarities
    """

    y_size = t_vect.shape[0]
    zeros = np.zeros((n_simulations + 1, y_size))

    zeros[0, :] = t_vect

    for i in range(n_simulations):
        zeros[i + 1, :] = np.random.choice(t_vect, y_size)

    return zeros.mean(axis=1)


def var_criterion(frequencies):
    """
    Value at Risk criterion
    """

    mu = np.mean(frequencies)
    std = np.std(frequencies)
    return norm.ppf(0.95, mu, std)


def lit_criterion(frequencies, u):
    """
        Littlewood's rule based criterion
    """

    mu = np.mean(frequencies)
    std = np.std(frequencies)

    return mu * (1 - norm.cdf(mu + u, mu, std))

