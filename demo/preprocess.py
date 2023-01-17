import pandas as pd


def preprocess(df):
    # Step 1
    df = df.drop(columns=['user_id'])
    # Step 2
    df['event_time_dt'] = df['event_time'].astype('datetime64[s]')
    df['event_time_ts']= df['event_time_dt'].view('int')
    df = df.drop(['event_time'],  axis=1)
    # Step 3
    df = df[df['user_session'].isnull()==False]

    # Step 4
    user_sessions = df.user_session.unique().tolist()
    user_session_to_index = {item: index for index, item in enumerate(user_sessions)}
    session_index_to_id = {index: item for index, item in enumerate(user_sessions)}
    df['user_session'] = df.user_session.map(user_session_to_index)

    # Step 5
    df = df.sort_values(['user_session', 'event_time_ts']).reset_index(drop=True)

    # Step 6
    df['product_id_past'] = df['product_id'].shift(1).fillna(0)
    df['session_id_past'] = df['user_session'].shift(1).fillna(0)
    df = df[~(
        (df['user_session'] == df['session_id_past'])
        & (df['product_id'] == df['product_id_past'])
    )]
    del(df['product_id_past'])
    del(df['session_id_past'])

    # Step 7
    session = df.groupby('user_session').size()
    session_df = df[df.user_session.isin(session[session > 1].index)].set_index('user_session')
    session_df = session_df.drop(columns=['category_id', 'event_time_ts'])

    return session_df, user_session_to_index, session_index_to_id


def get_product_index_map(session_df, start_with=0):
    """Create a mapping which map the original product id into token id.
    """
    products = session_df.product_id.unique().tolist()
    product_id_to_index = { item: index + start_with for index, item in enumerate(products) }
    product_index_to_id = { (index + start_with): item for index, item in enumerate(products) }
    return product_id_to_index, product_index_to_id
