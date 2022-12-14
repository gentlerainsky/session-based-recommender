import pandas as pd


def preprocess(df, train_from_date=None, evaluation_from_date='2021-01-01', test_from_date='2021-02-01'):
    """In this function, we preprocess the dataset by
    1. Remove `user_id`. This is done because we want to focus on session-based rather than sequential recommeder system.
    2. Preprocess date column from Python object to a type supported by Pandas
    3. Remove event with no user_session id.
    4. Label encode each user session as integer for easier computation.
    5. Sort the events by user_session and even_time
    6. Remove consecutive repeating event.
    7. Remove sessions with only 1 event left. Because we cannot use them for any training or testing.
    8. Split train/eval/test dataset by time period. This has a side effect that removes sessions that last over the split date.
    """

    # Step 1
    df = df.drop(columns=['user_id'])
    # Step 2
    df['event_time_dt'] = df['event_time'].astype('datetime64[s]')
    df['event_time_ts']= df['event_time_dt'].astype('int')
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
    session_df = df[df.user_session.isin(session[session > 1].index)]

    # Step 8
    session_max_event_time = session_df.groupby('user_session')['event_time_dt'].max().sort_values()
    session_min_event_time = session_df.groupby('user_session')['event_time_dt'].min().sort_values()
    min_max_session = pd.DataFrame({'start': session_min_event_time, 'end': session_max_event_time}).sort_values('end')

    if train_from_date is None:
        train_from_date = min_max_session.start.min()
    
    train_session_df = session_df[
        session_df.user_session.isin(min_max_session[
            (min_max_session.end >= pd.to_datetime(train_from_date))
            & (min_max_session.end < pd.to_datetime(evaluation_from_date))
        ].index)
    ].set_index('user_session')

    val_session_df = session_df[
        session_df.user_session.isin(min_max_session[
            (min_max_session.start >= pd.to_datetime(evaluation_from_date))
            & (min_max_session.end < pd.to_datetime(test_from_date))
        ].index)
    ].set_index('user_session')

    test_session_df = session_df[
        session_df.user_session.isin(min_max_session[min_max_session.start >= pd.to_datetime(test_from_date)].index)
    ].set_index('user_session')
    return train_session_df, val_session_df, test_session_df, user_session_to_index, session_index_to_id


def get_product_index_map(session_df, start_with=0):
    """Create a mapping which map the original product id into token id.
    """
    products = session_df.product_id.unique().tolist()
    product_id_to_index = { item: index + start_with for index, item in enumerate(products) }
    product_index_to_id = { (index + start_with): item for index, item in enumerate(products) }
    return product_id_to_index, product_index_to_id
