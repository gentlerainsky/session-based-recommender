from torch import nn
import torch


class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()

    def forward(self, logit):
        """This loss function is proposed by the following paper.
        Hidasi, Balázs, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk.
            “Session-Based Recommendations with Recurrent Neural Networks.”
            arXiv, March 29, 2016. http://arxiv.org/abs/1511.06939.
        """
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.sigmoid(diff).mean() + torch.sigmoid(logit ** 2).mean()
        return loss


class GRU4REC(nn.Module):
    """Implement the Gated Recurrent Neural Network.
    This model contains 3 main layers
    - Embedding layer: to learn feature representation of each item.
    - GRU layer: to learn the interaction of item viewing sequence from the user.
    - Linear Layer: to learn the probability of the next-item in the sequence.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 dropout_hidden=.5, dropout_input=0, batch_size=50, embedding_dim=-1, use_cuda=False):
        super(GRU4REC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.device = 'cuda' if use_cuda else 'cpu'
        self.fc = nn.Linear(hidden_size, output_size)
        self.final_activation = nn.ReLU()
        self.embedding = nn.Embedding(input_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        self = self.to(self.device)

    def forward(self, input, hidden):
        embedded = input.unsqueeze(0)
        embedded = self.embedding(embedded)
        output, hidden = self.gru(embedded, hidden)
        output = output.view(-1, output.size(-1))
        logit = self.final_activation(self.fc(output))
        return logit, hidden

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return h0
