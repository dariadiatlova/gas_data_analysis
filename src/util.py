from itertools import combinations
import numpy as np
import pandas as pd


def get_combination_idx(df: pd.DataFrame):
    """
    Function takes dataframe as input and returns an array of session trades pairs that were used for cluster analysis.
    :param df:
    :return: np.ndarray - 2D vector of shape 2 x num_combinations.
    """
    monthly_session_ids_combinations = np.array(list(combinations(df.session_id, 2)))
    return monthly_session_ids_combinations


def create_df_with_metric_values(metrics: np.ndarray, session_id: np.ndarray):
    """
    Function takes metrics and session_ids indices and returns a dataframe.
    :param metrics: np.ndarray of shape 3 x seq_len
    :param session_id:  np.ndarray of shape seq_len x 2
    :return: pd.DataFrame with columns: [cos_dist, mre, dtw, session_one_id, session_two_id]
    """
    cosine, mre, dtw = metrics[0, :], metrics[1, :], metrics[2, :]
    session_ids_comb_one = session_id[:, 0]
    session_ids_comb_two = session_id[:, 1]

    df = pd.DataFrame({'session_id_one': session_ids_comb_one,
                       'session_id_two': session_ids_comb_two,
                       'cos_dist': cosine,
                       'mre': mre,
                       'dtw': dtw})
    return df
