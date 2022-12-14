import torch
from torch.utils.data import Dataset
import numpy as np


class SessionDataset(Dataset):
    """This class provide some convenience function for the original dataset.
    """
    def __init__(self, event_df, product_id_to_index, sequence_length=30) -> None:
        self.event_df = event_df
        self.product_id_to_index = product_id_to_index
        self.event_df['product_index'] = self.event_df.product_id.map(product_id_to_index)
        self.event_df = self.event_df.dropna(subset=['product_index'])
        self.session_ids = self.event_df.index.unique()
        self.sequence_length = sequence_length

    def num_product(self):
        """Return number of (unique) products"""
        return len(self.product_id_to_index)

    def __len__(self):
        """Return number of (unqiue) sessions"""
        return self.session_ids.shape[0]

    def __getitem__(self, index):
        """Access data by session index
        """
        x = self.event_df.loc[self.session_ids[index]][-self.sequence_length-1:-1].product_index.values.tolist()
        y = self.event_df.loc[self.session_ids[index]].iloc[-1].product_index.values.tolist()
        return torch.tensor(x), torch.tensor(y)

    def get_session_length(self):
        """Compute cumulative summation of session length. This make it easier to access the beginning of each
        session in the event_df.
        """
        session_lengths = np.zeros(self.session_ids.shape[0] + 1, dtype=np.int32)
        session_lengths[1:] = self.event_df.groupby(level=0).size().cumsum().values
        return session_lengths
