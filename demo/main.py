import sys
import heapq
import torch
import random
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
import pandas as pd
from demo.preprocess import preprocess
from recsys.rnn.gru_model import GRU4REC
from recsys.preprocessor import load_product_id_mapper
from recsys.session_dataset import SessionDataset
from recsys.rnn.dataloader import RNNDataLoader

import warnings
warnings.filterwarnings('ignore')

## Loading Saved Data

def load_gru_model(checkpoint_path, product_id_to_index):
    """Load Deep Learning Model (GRU) from Pytorch Lightning Module
    state_dict to Pytorch Module
    """
    model = GRU4REC(
        input_size=len(product_id_to_index),
        hidden_size=100,
        output_size=len(product_id_to_index),
        embedding_dim=100,
        batch_size=50
    )
    model_dict = {}
    checkpoint = torch.load(checkpoint_path)
    for key in checkpoint['state_dict'].keys():
        k = '.'.join(key.split('.')[1:])
        model_dict[k] = checkpoint['state_dict'][key]
    model.load_state_dict(model_dict)
    model.eval()
    return model


def load_item_df():
    # use training dataset as a proxy for the list of all items
    train_df = pd.read_pickle('./data/train.pkl')
    item_df = train_df.reset_index()\
        .groupby('product_id')\
        .first()\
        .drop(columns=['user_session', 'event_type', 'event_time_dt', 'event_time_ts', 'category_id'])
    return item_df


def inference(model, hidden_state, batch):
    """Find the top N items for recommendation (Top 20 items in this task)
    """
    input, _, _ = batch
    logit, hidden_state = model(input, hidden_state)
    prediction = logit.detach().numpy()[0]
    top_n = heapq.nlargest(20, range(len(prediction)), prediction.take)
    return top_n, hidden_state


def model_recommend(model, demo_session_df, product_id_to_index, product_index_to_id):
    """Make prediction and ranking of the top items for recommendation.
    Because RNN is a seq model. It needs to run through all item in the session first
    before it can make a good prediction.
    """
    demo_dataset = SessionDataset(demo_session_df, product_id_to_index)
    demo_dataloader = RNNDataLoader(demo_dataset, batch_size=1)
    hidden_state = model.init_hidden(1)
    recommendation_steps = []
    for batch in demo_dataloader:
        top_n, hidden_state = inference(model, hidden_state, batch)
        recommendation = [product_index_to_id[idx] for idx in top_n]
        recommendation_steps.append(recommendation)
    return recommendation_steps


def print_table(console, data_df, next_product_id=None):
    """Pretty print a data frame.
    """
    table = Table(show_header=True, header_style="bold", show_footer=False)
    df = data_df
    if next_product_id is not None:
        df = data_df.reset_index()

    for column in df.columns:
        table.add_column(column)

    for index, row in df.iterrows():
        items = [str(i) for i in row.values]
        if (next_product_id is not None) and (row.product_id == next_product_id):
            items = [f"[bright_yellow]{i}[/bright_yellow]" for i in items]
        table.add_row(*items)
        table.box = box.MINIMAL
    console.print(table)


def parse_input(s):
    if s.lower() == 'q':
        return 'q'
    if s.isdigit():
        return int(s, 10)
    else:
        return None


def user_input():
    print("Input a session id (a number), Q to quit or press ENTER (or anything else) for a surprise!:")
    input_text = input()
    choice = parse_input(input_text)
    if choice == 'q':
        print('Thank you for your time!')
        sys.exit()
    elif choice is None:
        return None
    else:
        return choice


def loop(
    console,
    model,
    session_df,
    session_ids,
    product_id_to_index,
    product_index_to_id,
    item_df
    ):
    print_title(console)
    session_idx = user_input()
    while True:
        if session_idx is None:
            session_idx = random.randint(0, len(session_ids) - 1)

        demo_session_df = session_df.loc[session_ids[session_idx]].iloc[:-1]
        if demo_session_df.shape[0] <= 1:
            session_idx = None
            continue
        target = session_df.loc[session_ids[session_idx]].iloc[-1:]
        next_product_id = target.product_id.iloc[0]

        recommendation = recommend(
            model,
            demo_session_df,
            product_id_to_index,
            product_index_to_id,
            item_df
        )
        if recommendation is None:
            session_idx = None
            continue

        # next_item_ranks = recommendation[recommendation.product_id == next_product_id].index.tolist()
        print_output(console, session_idx, demo_session_df, recommendation, target, next_product_id)
        session_idx = user_input()


def print_title(console):
    console.clear()
    title_text = Text("TU Wien Applied Deep Learning Project: Session Recommendation System Demo")
    title_text.stylize("bold light_green")
    console.print(title_text)


def print_output(console, session_idx, session_df, recommendation, target, next_product_id):
    print_title(console)
    print(f'Recommendation for session id: {session_idx}')
    print('====================================')
    print('User Session')
    print('============')
    print_table(console, session_df)
    print('True Interaction')
    print('============')
    print_table(console, target)
    print('Recommendation list')
    print('===================')
    print_table(console, recommendation, next_product_id)
    text = Text("Yellow indicate the true next item of this session.")
    text.stylize("bold yellow", 0, 6)
    console.print(text)
    print('===================')


def recommend(model, demo_session_df, product_id_to_index, product_index_to_id, item_df):
    """Find top 20 item for recommendation.
    """
    recommendation_steps = model_recommend(
        model, demo_session_df, product_id_to_index, product_index_to_id
    )
    if len(recommendation_steps) == 0:
        return None
    recommendation = item_df.loc[recommendation_steps[-1]].reset_index()
    recommendation.index = recommendation.index + 1
    return recommendation


def main():
    console = Console()
    print_title(console)
    print()
    print('Loading Model and preparing data ...')
    checkpoint_path = 'saved_model/gru_model/lightning_logs/version_0/checkpoints/epoch=29-step=59490.ckpt'
    path = 'saved_model/gru_model/'
    product_id_to_index, product_index_to_id = load_product_id_mapper(path)
    model = load_gru_model(
        checkpoint_path=checkpoint_path,
        product_id_to_index=product_id_to_index
    )
    df = pd.read_csv('./demo/input/test_events.csv')
    session_df, _, _ = preprocess(df)
    session_ids = session_df.index.unique().values
    item_df = load_item_df()
    
    loop(
        console,
        model,
        session_df,
        session_ids,
        product_id_to_index,
        product_index_to_id,
        item_df
    )


if __name__ == '__main__':
    main()
