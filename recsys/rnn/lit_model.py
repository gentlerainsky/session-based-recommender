import torch
import pandas as pd
from pytorch_lightning import LightningModule
from recsys.rnn.gru_model import GRU4REC, TOP1Loss
from recsys.session_dataset import SessionDataset
from recsys.rnn.dataloader import RNNDataLoader
from recsys.preprocessor import get_product_index_map
from recsys.evaluation import evaluate


class LitGRUModel(LightningModule):
    def __init__(
        self,
        train_session_df,
        val_session_df,
        test_session_df,
        batch_size=50,
        top_k = 20,
        learning_rate = 1e-4
    ):
        super().__init__()
        self.loss_criterion = TOP1Loss()
        self.train_loss_log = []
        self.val_rank_log = []
        self.test_rank_log = []
        self.batch_size = batch_size
        self.train_session_df = train_session_df
        self.val_session_df = val_session_df
        self.test_session_df = test_session_df
        print(
            f"train_session_df.shape={self.train_session_df.shape}, "
            + f"val_session_df.shape={self.val_session_df.shape}, "
            + f"test_session_df.shape={self.test_session_df.shape}"
        )
        self.product_id_to_index, self.product_index_to_id = get_product_index_map(self.train_session_df)
        self.model = GRU4REC(
            input_size=len(self.product_id_to_index),
            hidden_size=100,
            output_size=len(self.product_id_to_index),
            embedding_dim=100,
            batch_size=self.batch_size
        )
        self.hidden_state = self.model.init_hidden()
        self.val_hidden_state = self.model.init_hidden(1)
        self.top_k = top_k
        self.learning_rate = learning_rate

    def forward(self, x, hidden):
        x = self.model(x, hidden)
        return x

    def reset_end_session_hidden(self, hidden, mark):
        if len(mark) != 0:
            hidden[:, mark, :] = 0
        return hidden

    def training_step(self, batch, batch_index):
        input, target, mark = batch
        input = input.to(self.device)
        target = target.to(self.device)
        # Resets hidden state for terminated sessions
        self.hidden_state = self.reset_end_session_hidden(self.hidden_state, mark)
        self.hidden_state = self.hidden_state.to(self.device).detach()
        logit, self.hidden_state = self(input, self.hidden_state)
        # Select the probability of the target items
        logit_sampled = logit[:, target.view(-1)]
        loss = self.loss_criterion(logit_sampled)
        return loss
    
    def validation_step(self, batch, batch_index):
        input, target, mark = batch
        input = input.to(self.device)
        target = target.to(self.device)
        self.val_hidden_state = self.reset_end_session_hidden(self.val_hidden_state, mark)
        self.val_hidden_state = self.val_hidden_state.to(self.device).detach()
        logit, self.val_hidden_state = self.model(input, self.val_hidden_state)
        logit_sampled = logit[:, target.view(-1)]
        loss = self.loss_criterion(logit_sampled)
        # Because the GRU learns from item to item instead of session to session,
        # We only calculate the evaluation metrics for the recommendation of the final item
        # of a session to make it comparable with other models.
        if len(mark) >= 0:
            recall, mrr = evaluate(logit[mark,:], target[mark], k=self.top_k)
            return loss, recall, mrr
        return loss, None, None

    def training_epoch_end(self, outs):
        losses = []
        for out in outs:
            loss = out['loss']
            losses.append(loss)
        self.train_loss_log.append(torch.mean(torch.stack(losses), dim=0).item())

    def validation_epoch_end(self, outs):
        losses = []
        recalls = []
        mrrs = []
        for out in outs:
            loss, recall, mrr = out
            losses.append(loss)
            # From the validation_step(), we calculate only when recal and mrr
            # are from the recommendation of the final item of a session.
            if recall is not None:
                recalls.append(recall)
            if mrr is not None:
                if torch.isnan(mrr):
                    mrrs.append(torch.tensor(0.0).to(self.device))
                else:
                    mrrs.append(mrr)
        self.val_hidden_state = self.model.init_hidden(1)
        
        self.val_rank_log.append({
            "loss": torch.mean(torch.stack(losses), dim=0).item(),
            "recall": torch.mean(torch.stack(recalls), dim=0).item(),
            "mrr": torch.mean(torch.stack(mrrs), dim=0).item(),
        })
        print(
            f"Validation Set (@epoch:{self.current_epoch}): loss={self.val_rank_log[-1]['loss']}, "
            + f"recall={self.val_rank_log[-1]['recall']}, "
            + f"mrr={self.val_rank_log[-1]['mrr']}"
        )

    def test_step(self, batch, batch_idx):
        input, target, mark = batch
        input = input.to(self.device)
        target = target.to(self.device)
        self.val_hidden_state = self.reset_end_session_hidden(self.val_hidden_state, mark)
        self.val_hidden_state = self.val_hidden_state.to(self.device).detach()
        logit, self.val_hidden_state = self.model(input, self.val_hidden_state)
        logit_sampled = logit[:, target.view(-1)]
        loss = self.loss_criterion(logit_sampled)
        if len(mark) >= 0:
            recall, mrr = evaluate(logit[mark,:], target[mark], k=self.top_k)
            return loss, recall, mrr
        return loss, None, None

    def test_epoch_end(self, outs):
        losses = []
        recalls = []
        mrrs = []
        for out in outs:
            loss, recall, mrr = out
            losses.append(loss)
            if recall is not None:
                recalls.append(recall)
            if mrr is not None:
                if torch.isnan(mrr):
                    mrrs.append(torch.tensor(0.0).to(self.device))
                else:
                    mrrs.append(mrr)
        
        self.test_rank_log.append({
            "loss": torch.mean(torch.stack(losses), dim=0).item(),
            "recall": torch.mean(torch.stack(recalls), dim=0).item(),
            "mrr": torch.mean(torch.stack(mrrs), dim=0).item(),
        })
        print(
            f"Testing Set loss={self.test_rank_log[-1]['loss']}, "
            + f"recall={self.test_rank_log[-1]['recall']}, "
            + f"mrr={self.test_rank_log[-1]['mrr']}"
        )
        self.val_hidden_state = self.model.init_hidden(1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = \
                SessionDataset(self.train_session_df, self.product_id_to_index)
            self.val_dataset = \
                SessionDataset(self.val_session_df, self.product_id_to_index)
        if stage == "test" or stage is None:
            self.test_dataset = \
                SessionDataset(self.test_session_df, self.product_id_to_index)

    def train_dataloader(self):
        return RNNDataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return RNNDataLoader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return RNNDataLoader(self.test_dataset, batch_size=1)
