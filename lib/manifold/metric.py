"""Methods to compute metrics."""


def compute_gaussian_similarity(distance, variance=1):
    """Compute Gaussian similarity

    Args:
        distance (numpy.ndarray(numpy.float)): Distance of shape [n_samples, ] between two points. Can be also used for
            whole numpy arrays for vectorization.
        variance (float): Variance of Gaussian Distribution.

    Returns:
        numpy.ndarray(numpy.float)): Similarity for each distance. Output shape is the same as input

    """
    # variance = 1 # According to paper the optimal value is found by hand, I'll try 1 for now
    similarity = (distance**2)/(2*variance**2)
    return similarity
