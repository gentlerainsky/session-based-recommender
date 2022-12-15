import torch
import random
import numpy as np


class TransformerDataLoader():
    """
    This data loader create data batch by
    1. stacking each session.
        For example, we have a `batch_size` of 3.
        [
            [93, 23, 45],
            [15, 24],
            [50, 99, 123, 145, 214]
        ]
    2. We left-pad each session with `padding_token` to have the same length of `max_sequence_length`.
        We trimmed longer sessions by removing older items.
        For example, with `max_sequence_length` = 4 and `padding_token` = 0
        [
            [0, 93, 23, 45],
            [0, 0,  15, 24],
            [99, 123, 145, 214]
        ]
    3. For input sequence, we replace the last item with `input_mask`.
        So, our model needs to predict the original sequence.
        For example,
        [
            [0, 93, 23, 1],
            [0, 0,  15, 1],
            [99, 123, 145, 1]
        ]
    4. We try masking more item in the sequence and let the model attempts
        to predict the original sequence.
        For example,
        From
        [ 5, 50, 99, 123, 145, 1 ]
        To
        [ 5, 1, 99, 123, 145, 1 ]
    """
    def __init__(self, dataset, max_sequence_length=10, batch_size=50, padding_token=0, random_mask_prob=0.2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.padding_token = padding_token
        self.input_mask = 1
        self.random_mask_prob = random_mask_prob

    def pad(self, values):
        """Left pad every session with `padding_token` until it has the length of `max_sequence_length`"""
        if len(values) < self.max_sequence_length + 1:
            pad_string = [self.padding_token] * (self.max_sequence_length + 1 - len(values))
            values = pad_string + values
        return values

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        df = self.dataset.event_df
        num_total_sessions = len(self.dataset)
        session_idxs = list(range(num_total_sessions))
        # Reshuffle the order of session. So we have a batch of random sessions.
        random.shuffle(session_idxs)
        finished = False
        idx = 0
        while not finished:
            if idx > len(session_idxs):
                break
            
            # Get the session information of the current batch
            batch_session_ids = [self.dataset.session_ids[i] for i in session_idxs[idx: idx+self.batch_size]]

            def get_product_list(df, is_random_mask):
                product_list = df.product_index.values
                if is_random_mask and self.random_mask_prob > 0:
                    if len(product_list) > 5:
                        # Randomly mask item in the seqeunce. This is in hope that the model
                        # can have something more to learn from.
                        if np.random.rand() > 0.5:
                            list_length = len(product_list)
                            indices = np.random.choice(
                                np.arange(list_length) - 1, size=int(np.floor((list_length - 1) * self.random_mask_prob)),
                                replace=False
                            ).astype(int)
                            product_list[indices] = self.input_mask
                product_list = self.pad(product_list.tolist())
                # Trim the session to limit the length of each sequence to be at most `max_sequence_length`
                # by removing older events.
                product_list = product_list[-(self.max_sequence_length):]
                return product_list

            rows = df.loc[batch_session_ids].groupby(level=0).apply(lambda df: get_product_list(df, False)).values.tolist()
            y = np.array(rows, dtype=np.long)
            
            rows = df.loc[batch_session_ids].groupby(level=0).apply(lambda df: get_product_list(df, True)).values.tolist()
            x = np.array(rows, dtype=np.long)
            # raise UnboundLocalError()
            # Mask the last item in the sequence
            x[:, -1] = self.input_mask
            yield torch.tensor(x.T), torch.tensor(y)
            idx += self.batch_size
