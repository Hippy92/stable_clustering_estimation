from sklearn.datasets import load_breast_cancer, make_circles, make_moons, make_blobs
from main import gaussian_samples
import matplotlib.pyplot as plt

data1, _ = make_blobs(n_samples=500, centers=4, n_features=2, random_state=3)
data2, _ = make_blobs(n_samples=500, centers=6, n_features=2, random_state=3)

data3, _ = gaussian_samples([(1, 1), (1, 4), (4, 4), (4, 1), (2.5, 2.5)], 100)

plt.scatter(data3[:, 0], data3[:, 1], c='green')
# plt.scatter(zero[:, 0], zero[:, 1], c='blue')
plt.show()
