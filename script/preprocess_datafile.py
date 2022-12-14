from recsys.preprocessor import preprocess
import pandas as pd
import pickle


if __name__ in '__main__':
    """This script preprocess and split the dataset into a train/val/test dataset.
    Then save the result as pickle files.
    """
    df = pd.read_csv("./data/events.csv")
    train_session_df, val_session_df, test_session_df, user_session_to_index, session_index_to_id = preprocess(df)
    train_session_df.to_pickle('./data/train.pkl')
    val_session_df.to_pickle('./data/val.pkl')
    test_session_df.to_pickle('./data/test.pkl')
    with open(r"./data/session_id_map.pkl", "wb") as output_file:
        pickle.dump(session_index_to_id, output_file)
