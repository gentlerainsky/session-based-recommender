import torch
from pytorch_lightning import LightningModule

from recsys.transformer.model import TransformerModel
from recsys.session_dataset import SessionDataset
from recsys.transformer.dataloader import TransformerDataLoader
from recsys.preprocessor import get_product_index_map
from recsys.evaluation import evaluate


class LitTransformerModel(LightningModule):
    def __init__(
        self,
        train_session_df,
        val_session_df,
        test_session_df,
        max_session_length=10,
        batch_size=50,
        learning_rate=0.0001
    ):
        super().__init__()
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.max_session_length = max_session_length
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
        self.start_with = 2
        self.product_id_to_index, self.product_index_to_id = get_product_index_map(self.train_session_df, start_with=self.start_with)
        self.model = TransformerModel(
            num_token=len(self.product_id_to_index) + self.start_with,
            d_model=150,
            nhead=5,
            d_hid=150,
            nlayers=5,
            pad_id=0,
            dropout=0.2
        )
        self.topk = 20
        self.learning_rate = learning_rate

    def forward(self, x, hidden):
        x = self.model(x, hidden)
        return x

    def training_step(self, batch, batch_idx):
        input, target = batch
        input = input.to(self.device)
        target = target.to(self.device)
        logits = self.model(input)
        # Calculate loss by reshape the `(sequence_length, batch, num_item)`
        # to `(sequence_length * batch, num_item)` which can be accepted by
        # the loss function. `target` is reshaped to be a vector of length
        # `sequence_length * batch` as well.
        loss = self.loss_criterion(
            logits.transpose(0, 1).reshape(-1, logits.transpose(0, 1).shape[-1]),
            target.reshape(-1)
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        input, target = batch

        input = input.to(self.device)
        target = target.to(self.device)
        logits = self.model(input)
        
        loss = self.loss_criterion(
            logits.transpose(0, 1).reshape(-1, logits.transpose(0, 1).shape[-1]),
            target.reshape(-1)
        )
        # The transformer model produce the whole sequence as an output.
        # But we only care about the last item of each sequence so we select
        # only the last one for evaluation.
        recall, mrr = evaluate(logits.transpose(0, 1)[:, -1, :], target[:, -1], k=self.topk)
        return loss, recall, mrr

    def training_epoch_end(self, outs):
        losses = []
        for out in outs:
            loss = out['loss']
            losses.append(loss)
        mean_loss = torch.mean(torch.tensor(losses)).item()
        self.train_loss_log.append(mean_loss)

    def validation_epoch_end(self, outs):
        losses = []
        recalls = []
        mrrs = []
        for out in outs:
            loss, recall, mrr = out
            losses.append(loss)
            recalls.append(recall)
            mrrs.append(mrr)
        self.val_rank_log.append({
            "loss": torch.mean(torch.tensor(losses, dtype=torch.float)).item(),
            "recall": torch.mean(torch.tensor(recalls, dtype=torch.float)).item(),
            "mrr": torch.mean(torch.tensor(mrrs, dtype=torch.float)).item(),
        })
        print(
            f"Validation Set (@epoch:{self.current_epoch}): "
            + f"loss={self.val_rank_log[-1]['loss']}, "
            + f"recall={self.val_rank_log[-1]['recall']}, "
            + f"mrr={self.val_rank_log[-1]['mrr']}"
        )
        
    def test_step(self, batch, batch_idx):
        input, target = batch

        input = input.to(self.device)
        target = target.to(self.device)
        
        logits = self.model(input)
        
        loss = self.loss_criterion(
            logits.transpose(0, 1).reshape(-1, logits.transpose(0, 1).shape[-1]),
            target.reshape(-1)
        )
        recall, mrr = evaluate(logits.transpose(0, 1)[:, -1, :], target[:, -1], k=self.topk)
        return loss, recall, mrr
    
    
    def test_epoch_end(self, outs):
        losses = []
        recalls = []
        mrrs = []
        for out in outs:
            loss, recall, mrr = out
            losses.append(loss)
            recalls.append(recall)
            mrrs.append(mrr)
        self.test_rank_log.append({
            "loss": torch.mean(torch.tensor(losses, dtype=torch.float)).item(),
            "recall": torch.mean(torch.tensor(recalls, dtype=torch.float)).item(),
            "mrr": torch.mean(torch.tensor(mrrs, dtype=torch.float)).item(),
        })
        print(
            f"Testing Set: loss={self.test_rank_log[-1]['loss']}, "
            + f"recall={self.test_rank_log[-1]['recall']}, "
            + f"mrr={self.test_rank_log[-1]['mrr']}"
        )

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
        return TransformerDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            max_sequence_length=self.max_session_length,
            random_mask_prob=0.2
        )

    def val_dataloader(self):
        return TransformerDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            max_sequence_length=self.max_session_length,
            random_mask_prob=0
        )

    def test_dataloader(self):
        return TransformerDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            max_sequence_length=self.max_session_length
        )
