import pandas as pd
import numpy as np


def date_time_reformat(df: pd.DataFrame):
    """
    Function takes dataframe as input with date and time columns and convert replace them with one timestamp column.
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """

    df['timestamp'] = pd.to_datetime(df.time + ' ' + df.date)
    df.drop(columns=['time', 'date'], inplace=True)
    return df


def filter_outliers(df: pd.DataFrame, only_monthly_data: bool=True):
    """
    Function takes a dataframe as input and filter all sessions that lasts more than 1 hour.
    Returned dataframe is indexed by session_ids.
    :param df: pd.DataFrame
    :param only_monthly_data: bool if true filter daily data from dataset
    :return: pd.DataFrame
    """

    session_ids = df.groupby('session_id')['timestamp'].apply(list).index.to_numpy()
    timestamps_collection = df.groupby('session_id')['timestamp'].apply(list).to_list()

    session_length_counter = lambda x: (max(x) - min(x)) / np.timedelta64(1, 'h')
    session_length_collection = np.array([session_length_counter(x) for x in timestamps_collection], dtype=np.float32)
    session_ids_to_drop = session_ids[session_length_collection > 1]

    df_idx = df.set_index('session_id')
    df_idx.drop(session_ids_to_drop, inplace=True)
    if only_monthly_data:
        df_idx = df_idx[df_idx.trading_type == 'monthly']

    return df_idx


def group_deals_by_session_id(df: pd.DataFrame):
    """
    Functions takes a dataframe and group all column values by session ids.
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    prices = df.groupby('session_id')['price'].apply(list).to_list()
    timestamps = df.groupby('session_id')['timestamp'].apply(list).to_list()
    lot_size = df.groupby('session_id')['lot_size'].apply(list).to_list()
    prices = [abs(np.array(prices[i])) / np.array(lot_size[i]) for i in range(len(prices))]
    deal_id = df.groupby('session_id')['deal_id'].apply(list).to_list()
    trading_type = df.groupby('session_id')['trading_type'].apply(list).to_list()
    trading_type = [np.unique(x)[0] for x in trading_type]
    platform_id = df.groupby('session_id')['platform_id'].apply(list).to_list()
    platform_id = [np.unique(x)[0] for x in platform_id]

    func = lambda x: [t.to_pydatetime().time().minute for t in x]
    deal_min = [func(x) for x in timestamps]

    df_deals_by_session_id = pd.DataFrame({'session_id': df.groupby('session_id')['price'].apply(list).index.to_list(),
                                  'deal_min': deal_min,
                                  'price': prices,
                                  'deal_id': deal_id,
                                  'trading_type': trading_type,
                                  'platform_id': platform_id})

    return df_deals_by_session_id.sort_values('session_id')
