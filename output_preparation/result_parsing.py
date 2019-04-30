import numpy as np

results = np.load('../results/result.npy')[()]

converted_results = dict()

for key, item in results.items():
    # key = (dataset, matrix distance, cluster number, clust alg name, noises level in %, criterion
    new_key = tuple(key[i] for i in [0, 1, 3, 4, 5])
    converted_results.setdefault(new_key, []).append((key[2], item))

for key, item in converted_results.items():
    converted_results[key] = np.array(item)

print(converted_results)




