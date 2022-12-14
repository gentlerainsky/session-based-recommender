import numpy as np
import torch


class RNNDataLoader():
    """
    This data loader create data batch as described by the following paper.
        Hidasi, Balázs, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk.
            “Session-Based Recommendations with Recurrent Neural Networks.”
            arXiv, March 29, 2016. http://arxiv.org/abs/1511.06939.
    A batch is construct by
    1. Order sessions by timestamp.
        For example, we have the following 5 sessions and a `batch_size` of 3.
        [
            [93, 23, 45],
            [15, 24],
            [50, 99, 123, 145],
            [54, 23, 99],
            [42, 1, 4]
        ]
    2. The first batch is a list of the first item from the first `batch_size` sessions.
        Ex.
        [
            93,
            15,
            50
        ]
    3. The next batch is the next items from the same sessions.
        Ex.
        [
            23,
            24,
            99
        ]
    4. We continue doing this until there is one session that we have reached the end.
    We continue by using the next unused session.
        Ex. The second session is exhausted. We use the fourth session instead.
        [
            45,
            54,
            123
        ]

    """
    def __init__(self, dataset, batch_size=50):
        self.dataset = dataset
        self.batch_size = batch_size

    # With the way the batch is construct.
    # The total number of batches is hard to determine.
    # def __len__(self):
    #     pass

    def __iter__(self):
        df = self.dataset.event_df
        # The cumulative summation of length of sessions
        session_length = self.dataset.get_session_length()
        session_indices = np.arange(self.batch_size, dtype=np.int32)
        # The start indices of each session in the event_df
        current = session_length[session_indices]
        # The end indices of each session in the event_df
        end = session_length[session_indices + 1]
        # The index of the last session
        max_session_indices = session_indices.max()
        
        # This list notes the exhausted sessions.
        mark = []

        finished = False
        while not finished:
            # The shortest number of the remaining items in the current session.
            min_length = (end - current).min().item()
            target_indices = df.iloc[current]['product_index'].values
            # Loop until one of the session ends.
            for i in range(min_length - 1):
                input_indices = target_indices
                # Use the next items as target
                target_indices = df.product_index.values[current + i + 1]
                input = torch.LongTensor(input_indices)
                target = torch.LongTensor(target_indices)
                yield input, target, mark

            current = current + (min_length - 1)
            # Determine which session has ended.
            mark = np.arange(len(session_indices), dtype=np.int32)[(end - current) <= 1].tolist()

            # For each ended session, replace it with the next available session.
            for i in mark:
                max_session_indices += 1
                if max_session_indices >= len(session_length) - 1:
                    finished = True
                    break
                session_indices[i] = max_session_indices
                current[i] = session_length[max_session_indices]
                end[i] = session_length[max_session_indices + 1]
