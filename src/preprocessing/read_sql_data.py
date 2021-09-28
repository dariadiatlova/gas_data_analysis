import pandas as pd
import sqlite3
from resources import DATA_ROOT


def get_df_from_sql():
    """
    Function returns a united dataframe of 2 tables: char_data and trading_session with unique deal_id's only.
    :return: Tuple[pd.dataframe]
    """
    con = sqlite3.connect(DATA_ROOT.parent / 'trade_info.sqlite3')
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")


    df_char_data = pd.read_sql_query("SELECT * from chart_data", con)
    df_trad_data = pd.read_sql_query("SELECT * from trading_session", con)
    con.close()

    df_trad_data.rename(columns={'id': 'session_id'}, inplace=True)

    df = df_char_data.merge(df_trad_data, on='session_id')
    df.drop_duplicates(subset=['deal_id'], inplace=True)
    return df
