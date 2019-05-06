from sklearn.datasets import make_circles, make_moons, make_blobs
from main import gaussian_samples
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

"""Only two dimensional data are plotted out"""


def plot_data(data, labels_, name):
    shape = data.shape
    zeroes = np.zeros((shape[0], shape[1] + 1))
    zeroes[:, :-1] = data
    zeroes[:, -1] = labels_

    # Colors and markers dicts
    color_map = defaultdict(lambda: 'purple', {0: 'green', 1: 'blue', 2: 'red', 3: 'brown', 4: 'cyan', 5: 'olive'})
    markers = defaultdict(lambda: 'bo', {0: "x", 1: "+", 2: "*", 3: "4", 4: "1", 5: "2"})

    for lab in set(labels_):
        plt.scatter(zeroes[zeroes[:, 2] == lab][:, 0], zeroes[zeroes[:, 2] == lab][:, 1],
                    c=color_map[lab], marker=markers[lab])

    plt.axis('off')
    plt.savefig('../results/plots/' + name + '.png')
    plt.close()


if __name__ == '__main__':
    databases = dict()
    labels = dict()

    databases['Moons'], labels['Moons'] = make_moons(700, noise=0.07)
    databases['Circles'], labels['Circles'] = make_circles(700, noise=0.06, factor=.5)
    databases['4 Gauss'], labels['4 Gauss'] = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=2)
    databases['6 Gauss'], labels['6 Gauss'] = make_blobs(n_samples=1000, centers=6, n_features=2, random_state=3)
    databases['5 Gauss'], labels['5 Gauss'] = gaussian_samples([(1, 1), (1, 4), (4, 4), (4, 1), (2.5, 2.5)], 200)

    for key, data_ in databases.items():
        plot_data(data_, labels[key], key)
