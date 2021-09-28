import numpy as np


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
from tslearn.metrics import dtw
from itertools import combinations


def compute_metrics(embeddings: np.ndarray):
    """
    Function computes the distances (mre, cosine and dtw) between all embeddings in the input array.
    :param embeddings: np.ndarray of shape num_embeddings x emb_size
    :return: np.ndarrary of shape 3 num_combinations,
    """

    embeddings_combinations = np.array(list(combinations(embeddings, 2)), dtype=np.float32)
    cosine_distances = np.array([cosine_similarity(
        embeddings_combinations[i, 0, :].reshape(-1, embeddings.shape[1]),
        embeddings_combinations[i, 1, :].reshape(-1, embeddings.shape[1]))[0][0]
                                 for i in range(embeddings_combinations.shape[0])])
    mre_distances = np.array([mean_absolute_error(embeddings_combinations[i, 0, :], embeddings_combinations[i, 1, :])
                              for i in range(embeddings_combinations.shape[0])])
    dtw_distances = np.array([dtw(embeddings_combinations[i, 0, :], embeddings_combinations[i, 1, :])
                              for i in range(embeddings_combinations.shape[0])])

    return np.stack([cosine_distances, mre_distances, dtw_distances])
