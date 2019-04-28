from simulations.clustering_iterations import *
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.datasets import load_breast_cancer, make_circles, make_moons, make_blobs
from collections import defaultdict
from sklearn import mixture
from sklearn.metrics import calinski_harabaz_score, silhouette_score
import json


def clust_alg(k):
    """
    Initialize the clustering algorithms
    :param k: number of clusters
    :return: dict of algorithms
    """
    clust_alg_init = dict()
    # clust_alg_init['Spectral Clustering'] = SpectralClustering(n_clusters=k, n_init=10, gamma=1.0, affinity="rbf",
    #                                                            n_neighbors=10, eigen_tol=0.0,
    #                                                            assign_labels="kmeans")
    clust_alg_init['Hierarchical Clustering'] = AgglomerativeClustering(n_clusters=k, affinity="euclidean")
    clust_alg_init['K-mean Clustering'] = KMeans(k, max_iter=500)
    clust_alg_init['EM'] = mixture.GMM(n_components=k)

    return clust_alg_init


def gaussian_samples(means, size_per_sample):
    center_number = len(means)
    zeroes = np.zeros([center_number * size_per_sample, 2])
    cov = [[0.00, 0.00], [0.1, 0.1]]
    for idx, mean in enumerate(means):
        zeroes[idx*size_per_sample:(idx + 1)*size_per_sample, :] = \
            np.random.multivariate_normal(mean, cov, size_per_sample)

    return zeroes


def data_sets():
    """
    The function returns the datasets for experiments
    :return dict() of datasets:
    """
    databases = dict()
    databases['Iris'] = load_iris()['data']
    databases['Wine'] = load_wine()['data']
    databases['Cancer'] = load_breast_cancer()['data']
    databases['Moons'], _ = make_moons(500, noise=0.06)
    databases['Circles'], _ = make_circles(500, noise=0.06, factor=.5)
    databases['4 Gauss'], _ = make_blobs(n_samples=500, centers=4, n_features=2, random_state=0)
    databases['6 Gauss'], _ = make_blobs(n_samples=500, centers=6, n_features=2, random_state=0)

    return databases


def distances_f():
    return {'cosine': cosine_distance, 'jaccar': jaccar_distance, 'm_dist': m_distance, 'lb_dist': lb_distance}


def criteria():
    return {'VaR': var_criterion, 'Expected_changes': lit_criterion}


def indices_fun(data_in, labels_):
    index_set = dict()
    index_set['Calinski_Harabaz'] = calinski_harabaz_score(data_in, labels_)
    index_set['Silhouette'] = silhouette_score(data_in, labels_)

    return index_set


if __name__ == '__main__':
    errors_perc = [0.01, 0.07]  # Errors is 1% and 7% from average value
    clusters_n = [k for k in range(2, 11)]  # Number of clusters from 2 to 10
    perturbations_n = 100  # Perturbations times
    boot_n = 1000  # Bootstrap iterations
    results = defaultdict(defaultdict)  # Collect results
    indices = dict()  # Collect results
    for data_key, data in data_sets().items():
        for clust_n in clusters_n:
            for alg_key, clustering_alg in clust_alg(clust_n).items():
                # Original data clustering
                clusters_orig = clustering_simulation(data, clustering_alg)

                if hasattr(clustering_alg, 'labels_'):
                    labels = clustering_alg.labels_
                else:
                    labels = clustering_alg.predict(data)
                indices[data_key, clust_n, alg_key] = indices_fun(data, labels)
                print(indices)
                for error in errors_perc:
                    # Noised samples generation
                    noises_samples = [generate_noised_sample(data, error) for i in range(perturbations_n)]
                    # Similarity matrix calculation
                    sim_matrices = [clustering_simulation(data_i, clustering_alg) for data_i in noises_samples]
                    for d_key, dist_f in distances_f().items():
                        # Similarities with original matrix C
                        sim = similarities_calc(sim_matrices, clusters_orig, dist_f)
                        # Result bootstrap
                        boot_fr = bootstrap(boot_n, np.array(sim))
                        # Calculate criterion
                        results[data_key, d_key, clust_n, alg_key, error, 'VaR'] = var_criterion(boot_fr)
                        results[data_key, d_key, clust_n, alg_key, error, 'Expected_frequency_0.01'] = lit_criterion(
                            boot_fr, 0.01)
                        results[data_key, d_key, clust_n, alg_key, error, 'Expected_frequency_0.001'] = lit_criterion(
                            boot_fr, 0.001)
                        results[data_key, d_key, clust_n, alg_key, error, 'Expected_frequency_0.005'] = lit_criterion(
                            boot_fr, 0.005)
                        results[data_key, d_key, clust_n, alg_key, error, 'Expected_frequency_0'] = lit_criterion(
                            boot_fr, 0.0)
                        print(results)

    np.save('./results/result.npy', results)
    np.save('./results/indices.npy', indices)

    # data = load_iris()['data']
    # # data = load_breast_cancer()['data']
    # # data = data / data.max(axis=0)
    #
    # set_dt, lb = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0,)
    # print(set_dt)
    # ones = set_dt[[val == 1 for i, val in enumerate(lb)]]
    # zero = set_dt[[val == 0 for i, val in enumerate(lb)]]
    # ones = gaussian_samples([[1, 1], [4, 4], [1, 4], [4, 1]], 100)
    # ones, _ = make_blobs(n_samples=1000, centers=6, n_features=2, random_state=0)
    # plt.scatter(ones[:, 0], ones[:, 1], c='green')
    # # plt.scatter(zero[:, 0], zero[:, 1], c='blue')
    # plt.show()
    #
    # spectarl = SpectralClustering(n_clusters=2, n_init=10, gamma=1.0, affinity="rbf", n_neighbors=10, eigen_tol=0.0,
    #                               assign_labels="kmeans")
    # hierarchicle = AgglomerativeClustering(n_clusters=2, affinity="euclidean")
    # k_mean = KMeans(3, max_iter=500)
    # clust_alg = k_mean
    # clusters_orig = clustering_simulation(data, clust_alg)
    # noises_samples = [generate_noised_sample(data, 1) for i in range(10)]
    # sim_matrices = [clustering_simulation(data_i, clust_alg) for data_i in noises_samples]
    # sim = similarities_calc(sim_matrices, clusters_orig, m_distance)
    # boot_fr = bootstrap(100, np.array(sim))
    # crit = lit_criterion(boot_fr, 0.005)
    # print({1: crit})
    # with open('./results/result.json', 'w') as fp:
    #     json.dump({1: crit}, fp)


