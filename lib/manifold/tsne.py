"""TSNE class implementation."""
import numpy as np
import tqdm

from sklearn.neighbors import NearestNeighbors

from lib.manifold.metric import compute_gaussian_similarity
from lib.manifold.base_class import ManifoldBase


class TSNE(ManifoldBase):
    """TSNE class.

    For detailed description reder to T-SNE paper (http://www.cs.toronto.edu/~hinton/absps/tsne.pdf)

    """

    def __init__(self,
                 seed=42,
                 n_components=2,
                 perplexity=10,
                 n_iter=200,
                 learning_rate=0.2,
                 fine_tune_share=0.1):
        """Init.

        Args:
            seed (int): Random seed to ensure reproducibility.
            n_components (int): Number of components of output space.
            perplexity (int): Number of nearest neighbours considered during optimization.
            n_iter (int): Number of iterations.
            learning_rate (float): Learning rate.
            fine_tune_share (float): Share of fine-tuning epochs. Example: If fine_tune_share is 0.1
                and n_iter is 100, than 100*(1-fine_tune_share)= 90 are trained with the original
                learning rate, the rest 0.1*100=10 epochs are trained with 0.1*learning_rate

        """
        self._random_state = np.random.RandomState(seed)
        self._n_components = n_components
        self._perplexity = perplexity
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._fine_tune_share = fine_tune_share
        self._fine_tune_epoch = int(self._n_iter*(1-self._fine_tune_share))

    def fit(self, data):
        """Perform TSNE on data.

        Args:
            data (numpy.ndarray(numpy.float)): Data to fit of shape [n_samples, n_features].

        Returns:
            numpy.ndarray(numpy.float): Data in embedded space of shape
                [n_samples, self._n_components]

        """
        n_samples = data.shape[0]

        # Init data into embedded space
        data_embedded = self._random_state.randn(n_samples, self._n_components)

        # Compute asymmetric probablity function in original/old space
        neighbour_distances, neighbour_idx = \
            NearestNeighbors(n_neighbors=self._perplexity,
                             metric='euclidean').fit(data).kneighbors(data)

        asym_prob_table_old_space = _compute_asym_prob_table(
            similarity_function=compute_gaussian_similarity,
            neighbour_distances=neighbour_distances,
            neighbour_idx=neighbour_idx)

        data_embedded = self._gradient_descent(asym_prop_old_space=asym_prob_table_old_space,
                                               data=data_embedded,
                                               neighbours_old_space=neighbour_idx)

        return data_embedded

    def _gradient_descent(self,
                          asym_prop_old_space,
                          data,
                          neighbours_old_space):
        """Optimize the model.

        Args:
            asym_prop_old_space (numpy.ndarray(numpy.float)): Table of size [n_samples, n_samples],
                where each cell shows the asymmetic probabilty between to samples (indices refer to
                whatsamples) of the orignal data.
            data (numpy.ndarray(numpy.float)): Data of new space. Data should have shape of
                [n_samples, n_features].
            neighbours_old_space (numpy.ndarray(numpy.float): Idx in shape [n_samples, n_neighbours]
                to the neighbours, for calculation example refer to from sklearn.neighbors import
                NearestNeighbors.

        Returns:
            numpy.ndarray(numpy.float): Data embedded in new space. Data shape is
                [n_samples, self._n_components]

        """

        neighbour_distances, neighbour_idx_table = \
            NearestNeighbors(n_neighbors=self._perplexity,
                             metric='euclidean').fit(data).kneighbors(data)

        asym_prob_new_space = _compute_asym_prob_table(
            similarity_function=compute_gaussian_similarity,
            neighbour_distances=neighbour_distances,
            neighbour_idx=neighbour_idx_table)

        learning_rate = self._learning_rate

        for iter_idx in tqdm.tqdm(range(self._n_iter), desc="Fit data with TSNE"):
            for first_idx in range(data.shape[0]):
                sum_value = 0

                neighbour_table = np.concatenate([neighbours_old_space, neighbour_idx_table],
                                                 axis=1)

                for neighbour_idx in range(neighbour_table.shape[1]):

                    second_idx = neighbour_table[first_idx, neighbour_idx]
                    sum_value += \
                        2 * ((data[first_idx] - data[second_idx]) *
                             (asym_prop_old_space[first_idx, second_idx] -
                              asym_prob_new_space[first_idx, second_idx] +
                              asym_prop_old_space[second_idx, first_idx] -
                              asym_prob_new_space[second_idx, first_idx]))

                data[first_idx] -= learning_rate * sum_value

            if iter_idx % 5 == 0:
                neighbour_distances, neighbour_idx_table = \
                    NearestNeighbors(n_neighbors=self._perplexity,
                                     metric='euclidean').fit(data).kneighbors(data)

                asym_prob_new_space = _compute_asym_prob_table(
                    similarity_function=compute_gaussian_similarity,
                    neighbour_distances=neighbour_distances,
                    neighbour_idx=neighbour_idx_table)

            if iter_idx == self._fine_tune_epoch:
                learning_rate *= 0.1

        data -= np.mean(data)
        data /= np.std(data)
        return data


def _compute_asym_prob_table(neighbour_distances,
                             neighbour_idx,
                             similarity_function=compute_gaussian_similarity):
    """Compute asymetric probability table.

    For details refer to the T-SNE paper.

    Args:
        neighbour_distances (numpy.ndarray(numpy.float)): Distances in shape
        [n_samples, n_neighbours]. For calculation example refer to from sklearn.neighbors import
            NearestNeighbors.
        neighbour_idx (numpy.ndarray(numpy.float)): Idx in shape [n_samples, n_neighbours] to the
        neighbours, for example calculation refer to from sklearn.neighbors import NearestNeighbors.
        similarity_function (function): Function to call calculate the distance from. For example
            refer to manifold.metric.compute_gaussian_similarity().

    Returns:
        numpy.ndarray(numpy.float): Table of size [n_samples, n_samples], where each cell shows the
            asymmetric probability between to samples (indices refer to what samples).

    """
    dissim_table = _compute_dissim_table(similarity_function=similarity_function,
                                         neighbour_idx=neighbour_idx,
                                         neighbour_distances=neighbour_distances)
    asym_similarity_table = _compute_asym_prob_table_given_dissim(similarity_table=dissim_table,
                                                                  neighbour_idx_table=neighbour_idx)

    return asym_similarity_table


def _compute_dissim_table(neighbour_idx,
                          neighbour_distances,
                          similarity_function=compute_gaussian_similarity):
    """Compute dissimilarity table.

    For details refer to the T-SNE paper.


    Args:
        neighbour_distances (numpy.ndarray(numpy.float)): Refer to
            manifold.tsne._compute_asym_prob_table() docstring.
        neighbour_idx(numpy.ndarray (numpy.float)):  Refer to
            manifold.tsne._compute_asym_prob_table() docstring.
        similarity_function (function):  Refer to manifold.tsne._compute_asym_prob_table() docstring

    Returns:
        numpy.ndarray(numpy.float): Table of size [n_samples, n_samples], where each cell shows the
            dissimilarity between to samples (indices refer to what samples).

    """
    n_samples = neighbour_idx.shape[0]
    table = np.zeros([n_samples, n_samples])
    skip_table = np.zeros_like(table, dtype=np.bool)

    for first_idx in range(n_samples):
        for second_idx in range(neighbour_idx.shape[1]):
            if first_idx != second_idx and not skip_table[first_idx][second_idx]:
                skip_table[first_idx][second_idx] = True
                skip_table[second_idx][first_idx] = True

                similarity = similarity_function \
                    (distance=neighbour_distances[first_idx, second_idx])

                table[first_idx][second_idx] = similarity
                table[second_idx][first_idx] = similarity

    return table


def _compute_asym_prob_table_given_dissim(similarity_table, neighbour_idx_table):
    n_samples = similarity_table.shape[0]
    asym_similarity_table = np.zeros([n_samples, n_samples])

    for idx in range(n_samples):
        neighbours_idx = neighbour_idx_table[idx, :]
        denom = np.sum(np.exp(-similarity_table[idx, neighbours_idx]))

        for neighbour_idx in neighbours_idx:
            asym_similarity_table[idx, neighbour_idx] = \
                np.exp(-similarity_table[idx, neighbour_idx]) / denom

    return asym_similarity_table
