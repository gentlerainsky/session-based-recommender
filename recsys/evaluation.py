import torch


def get_recall(indices, targets):
    """
    This metric (Recall@K) counts whether the `next-item` in the session is within top K recommended items or not.
    """
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return torch.tensor(0.0)
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = torch.tensor(float(n_hits)) / targets.size(0)
    return recall


def get_mrr(indices, targets): #Mean Receiprocal Rank --> Average of rank of next item in the session.
    """
    This metric (MRR@K) calculate the mean of the receiprocal rank of each sessions.
    The Rank refers to the rank of `next-item` in the session in the top K recommended items.
    The receiprocal rank is calculated by 1/Rank.
    """
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr


def evaluate(indices, targets, k=20):
    """Evaluates the model using Recall@K, MRR@K scores.
    """
    _, indices = torch.topk(indices, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    return recall, mrr
