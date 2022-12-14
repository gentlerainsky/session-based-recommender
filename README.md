# Session Based Recommender System

Name: Thummanoon Kunanuntakij

Student ID: 12122522

---

## Assignment 2 - Hacking

### Next Item Prediction/Recommendation Task in Session Based Recommender System

In this project, the session-based recommender system task is framed into next-item prediction
which is similar to next-word prediction in NLP.

### Evaluation Metric
In the evaluation step, the machine learning models are tested by giving the sequence of all items except the last one from each session.
The models have to predict the ranking of each item.

There are 2 metrics used.
- Recall@20 - The percentage of the next-item appears in the top-20 ranking.
- MRR@20 (Mean Reciprocal Rank) - It is the reciprocal of the ranking of the next-item in the top-20 ranking (1/Rank) averaged over the batch. If it is not in the rank, the value is 0.

Both values are calculated by feeding the most recent subset of the data. The validation set contains data from January 2021, and the testing set includes data from February 2021.

### Target Evaluation Metric

Target evaluation metrics are the following.
- Recall@20 > 0.2
- MRR@20 > 0.2

In other words, the recommended item should be included in the Top 5 suggestions from the system.

### Achieved Evaluation Metric

In valuation set (Sessions data in January 2021)

| Model        | Recall@20  | MRR@20 |
| -------------| -----------|--------|
| RNN with GRU | 0.3115     | 0.1317 |
| Transformer  | 0.0664     | 0.0169 |

In testing set (Session data in February 2021)

| Model        | Recall@20  | MRR@20 |
| -------------| -----------|--------|
| RNN with GRU | 0.2922     | 0.1242 |
| Transformer  | 0.0596     | 0.0153 |

### Work Breakdown

- (~20 Hours) Clean up, structure, and establish a workflow to train and test the model on the chosen dataset.
    - Initially, I tried to make a different recommender task: purchase-item prediction.
    But the quality of the dataset I picked wasn't as good as I assumed. After spending time cleaning and trying to reformat it the fit the original task, there were only a thousand sessions left which were not enough for the training.
    - After reviewing the literature again, I realized that most papers made only next-item predictions. It surprised me because most papers used standard datasets like `YOOCHOOSE` and `Diginetica.` Both were originally created for purchase-item prediction.
    - In the end, I changed my focus to making next-item predictions instead because of both the limitation of the dataset and it was what my reference papers did.
- (~20 Hours) Review & Implement generic deep learning methods as baselines ([RNN]).
    - I implement RNN using the idea from Hidasi, Balázs, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. "Session-Based Recommendations with Recurrent Neural Networks." arXiv, March 29, 2016. [http://arxiv.org/abs/1511.06939](http://arxiv.org/abs/1511.06939).
    - Most of the time was spent on debugging the training and evaluating the model.
- (~20 Hours) Implementing the ~~state-of-the-art architecture ([Transformers4Rec])~~.
    - In [Transformers4Rec], they develop a framework to utilize existing transformer architecture from HuggingFace. I didn't follow the paper as they are mainly about the engineering of the framework rather than the deep learning architecture.
    - I chose to implement only a vanilla Transformer using a module from Pytorch to learn how to utilize transformer architecture for this task.
    The Transformer architecture 
    - Most of the time was spent on debugging the training and evaluating the model.
- (~5 Hours) Optimizing promising models.
    - I optimized the model manually as I didn't have enough time left because of debugging.

### Run

- `./script/preprocess_datafile.py` - Preprocess and train/val/test split the dataset.
- `./gru.ipynb` - Notebook for the RNN with GRU Model.
- `./transformer.ipynb` - Notebook for the Transformer Model.

