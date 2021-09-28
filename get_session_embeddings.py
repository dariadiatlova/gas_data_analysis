import pandas as pd
import numpy as np


def get_session_embeddings(df: pd.DataFrame, dim: int = 2, daily: bool = False):
    """
    Function takes pandas dataframe with session ids and other values and returns numpy array of shape (nb_sessions,
    60).
    :param df pd.DataFrame
    :param dim int - number of with data to use for embedding creation
    :param daily bool - if True the length of embedding is set to 30, (60 otherwise)
    :return: Tuple[
                    - np.ndarray of shape (nb_sessions, 60),
                    - df with columns [session_ids, session_embedding]
                    ]
    """
    if daily:
        time_length = 30
    else:
        time_length = 60
    session_embeddings = np.zeros([df.shape[0], time_length], dtype=np.float32)
    last_session_val = df.iloc[0, dim][0]
    column_name = df.columns[dim]

    for i, session in enumerate(df.session_id):
        for minute in range(time_length):
            if minute in df[df.session_id == session].deal_min.to_list()[0]:
                for j, val in enumerate(df[df.session_id == session].deal_min.to_list()[0]):
                    if val == minute:
                        last_session_val = df[df.session_id == session][column_name].to_list()[0][j]
                        break
            session_embeddings[i][minute] = last_session_val

    rows_min = np.clip(np.min(session_embeddings, axis=1), -np.inf, 0)
    rows_min = np.repeat(rows_min, time_length).reshape(len(rows_min), -1)
    norms = np.clip(np.linalg.norm(session_embeddings + abs(rows_min), axis=1), 1e-8, np.inf)
    session_embeddings = (session_embeddings + abs(rows_min)) / np.repeat(norms, time_length).reshape(len(rows_min), -1)

    df = pd.DataFrame({'session_id': df.session_id, 'session_embedding': df.session_id})
    df['session_embedding'] = pd.Series([i for i in session_embeddings], index=df.index)
    return session_embeddings, df
