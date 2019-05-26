# TODO Docstring me

import numpy as np
import tqdm

from sklearn.neighbors import NearestNeighbors
""""""


class TSNE: # TODO BaseClass me

    def __init__(self, seed=42, n_components=2, perplexity=10, n_iter=200, learning_rate=0.2):
        self._random_state = np.random.RandomState(seed)
        self._n_components = n_components
        self._perplexity = perplexity
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._fine_tune_share = 0.1
        self._fine_tune_epoch = int(self._n_iter*(1-self._fine_tune_share))

    def fit(self, data):

        n_samples = data.shape[0]

        # Init
        data_embedded = self._random_state.randn(n_samples, self._n_components)

        # Compute
        neighbour_distances, neighbour_idx = \
            NearestNeighbors(n_neighbors=self._perplexity, metric='euclidean').fit(data).kneighbors(data)

        asym_prob_table_old_space = compute_asym_prob_table(similarity_function=compute_gaussian_similarity,
                                                            neighbour_distances=neighbour_distances,
                                                            neighbour_idx=neighbour_idx)

        data_embedded = self.gradient_descent(asym_prop_old_space=asym_prob_table_old_space,
                                              data=data_embedded,
                                              neighbours_old_space=neighbour_idx)

        return data_embedded

    def gradient_descent(self,
                         asym_prop_old_space,
                         data,
                         neighbours_old_space):

        neighbour_distances, neighbour_idx_table = \
            NearestNeighbors(n_neighbors=self._perplexity, metric='euclidean').fit(data).kneighbors(data)

        asym_prob_new_space = compute_asym_prob_table(similarity_function=compute_gaussian_similarity,
                                                      neighbour_distances=neighbour_distances,
                                                      neighbour_idx=neighbour_idx_table)

        learning_rate = self._learning_rate

        for iter_idx in tqdm.tqdm(range(self._n_iter), desc="Fit data with TSNE"):
            for first_idx in range(data.shape[0]):
                sum_value = 0

                neighbour_table = np.concatenate([neighbours_old_space, neighbour_idx_table], axis=1)

                for neighbour_idx in range(neighbour_table.shape[1]):

                    second_idx = neighbour_table[first_idx, neighbour_idx]
                    sum_value += \
                        2 * ((data[first_idx] - data[second_idx]) *
                             (asym_prop_old_space[first_idx, second_idx] - asym_prob_new_space[first_idx, second_idx] +
                              asym_prop_old_space[second_idx, first_idx] - asym_prob_new_space[second_idx, first_idx]))

                data[first_idx] -= learning_rate * sum_value

            if iter_idx % 5 == 0:
                neighbour_distances, neighbour_idx_table = \
                    NearestNeighbors(n_neighbors=self._perplexity, metric='euclidean').fit(data).kneighbors(data)

                asym_prob_new_space = compute_asym_prob_table(similarity_function=compute_gaussian_similarity,
                                                              neighbour_distances=neighbour_distances,
                                                              neighbour_idx=neighbour_idx_table)

            if iter_idx == self._fine_tune_epoch:
                learning_rate *= 0.1

        data -= np.mean(data)
        data /= np.std(data)
        return data


def compute_asym_prob_table(similarity_function, neighbour_distances, neighbour_idx):
    dissim_table = compute_dissim_table(similarity_function=similarity_function,
                                        neighbour_idx=neighbour_idx,
                                        neighbour_distances=neighbour_distances)
    asym_similarity_table = _compute_asym_prob_table(similarity_table=dissim_table,
                                                     neighbour_idx_table=neighbour_idx)

    return asym_similarity_table


def compute_dissim_table(similarity_function, neighbour_idx, neighbour_distances):
    n_samples = neighbour_idx.shape[0]
    table = np.zeros([n_samples, n_samples])
    skip_table = np.zeros_like(table, dtype=np.bool)

    for first_idx in range(n_samples):
        for second_idx in range(neighbour_idx.shape[1]):
            if first_idx != second_idx and not skip_table[first_idx][second_idx]:
                skip_table[first_idx][second_idx] = True
                skip_table[second_idx][first_idx] = True

                similarity = similarity_function(distance=neighbour_distances[first_idx, second_idx])

                table[first_idx][second_idx] = similarity
                table[second_idx][first_idx] = similarity

    return table


def compute_gaussian_similarity(distance, variance=10):
    # variance = 1 # According to paper the optimal value is found by hand, I'll try 1 for now
    similarity = (distance**2)/(2*variance**2)
    return similarity


def _compute_asym_prob_table(similarity_table, neighbour_idx_table):
    n_samples = similarity_table.shape[0]
    asym_similarity_table = np.zeros([n_samples, n_samples])

    for idx in range(n_samples):
        neighbours_idx = neighbour_idx_table[idx, :]
        denom = np.sum(np.exp(-similarity_table[idx, neighbours_idx]))

        for neighbour_idx in neighbours_idx:
            asym_similarity_table[idx, neighbour_idx] = np.exp(-similarity_table[idx, neighbour_idx]) / denom

    return asym_similarity_table
