# Session Based Recommender System

Name: Thummanoon Kunanuntakij

Student ID: 12122522

---

*1. References to at least two scientific papers that are related to your topic*

[Survey Paper]

Wang, Shoujin, Longbing Cao, Yan Wang, Quan Z. Sheng, Mehmet Orgun, and Defu Lian. “A Survey on Session-Based Recommender Systems.” arXiv, May 15, 2021. [http://arxiv.org/abs/1902.04864](http://arxiv.org/abs/1902.04864).

[RNN]

Hidasi, Balázs, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. “Session-Based Recommendations with Recurrent Neural Networks.” arXiv, March 29, 2016. [http://arxiv.org/abs/1511.06939](http://arxiv.org/abs/1511.06939).

[GNN]

Wu, Shu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan. “Session-Based Recommendation with Graph Neural Networks.” *Proceedings of the AAAI Conference on Artificial Intelligence* 33 (July 17, 2019): 346–53. [https://doi.org/10.1609/aaai.v33i01.3301346](https://doi.org/10.1609/aaai.v33i01.3301346).

[Transformers4Rec]

Souza Pereira Moreira, Gabriel de, Sara Rabhi, Jeong Min Lee, Ronay Ak, and Even Oldridge. “Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation.” In *Fifteenth ACM Conference on Recommender Systems*, 143–53. Amsterdam Netherlands: ACM, 2021. [https://doi.org/10.1145/3460231.3474255](https://doi.org/10.1145/3460231.3474255).

[SGNN-HN]

Pan, Zhiqiang, Fei Cai, Wanyu Chen, Honghui Chen, and Maarten de Rijke. “Star Graph Neural Networks for Session-Based Recommendation.” In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management*, 1195–1204. Virtual Event Ireland: ACM, 2020. [https://doi.org/10.1145/3340531.3412014](https://doi.org/10.1145/3340531.3412014).

[TAGNN]

Yu, Feng, Yanqiao Zhu, Qiang Liu, Shu Wu, Liang Wang, and Tieniu Tan. “TAGNN: Target Attentive Graph Neural Networks for Session-Based Recommendation.” In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 1921–24, 2020. https://doi.org/10.1145/3397271.3401319.

*2. A decision of a topic of your choice (see below for inspiration)*

**Session Based Recommender System**

*3. A decision of which type of project you want to do (see below)*

**Bring your own method**

*4. A written summary that should contain:*

*a. Short description of your project idea and the approach you intend to use*

A session-Based Recommender System is a task in a recommender system where a system tries to recommend an item to an unknown user based on the list of items that the user views or interacts with during a session. This problem differs from a conventional recommender system in the way that the system doesn’t aware of information about their background (long-term preference). This task is practical as it shows up in all e-commerce websites when a random user visit a website (or before a member signs in). Not only its practical usage, but this task is also interesting in its technical part. Many machine learning methods can be applied in an attempt to solve the task including traditional methods like collaborative filtering and Markov chain, as well as recent deep learning methods such as recurrent neural networks, transformer, and graph neural networks.

This project will implement or reuse at least 2 models. The first models is a recurrent neural network. These model will serve as a baseline. Then, I will try to improve the result by implementing a graph neural network or a transformer model [Transformers4Rec]. Finally, I will try to optimize them to see if I can improve the result by utilize more meta data from the dataset, optimizing hyper parameter or implementing the state-of-the-art architectures ([SGNN-HN] and [TAGNN]).

For the dataset, conventionally the benchmarking dataset used in the literature are competition data. In this project, I will use a more realistic dataset found in Kaggle which contains more item metadata and detailed user action sequence such as multiple purchases in a session and remove-from-cart action. It is interesting to see how each model perform a real world dataset.

For the demo, it is not easy to develop one for this task as the dataset doesn’t contain rich product information such as description and photo. Thus, I cannot create a realistic demo like an e-commerce page. Nevertheless, I aim to show how the recommended products change over time after each actions performed by a user.

*b. Description of the dataset you are about to use (or collect)*

Dataset: [eCommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) or [eCommerce events history in electronics store](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store)

Description:

This is a real world dataset. The dataset is tabular containing 7 months of data (from October 2019 to April 2020) from online stores. A suitable subset data of the chosen dataset will be used.

For each row, it contains product information (such as product category, brand and price), user information (such as user id and session) and also user actions. A unique characteristic of this dataset in comparison with other conventional one is in the detail of user actions. It recorded a realistic sequence of action for each session. Normally, in other datasets, they recorded only a sequence of viewed items and end a session with a purchase. In this dataset, it noted all the journey from viewing, adding and removing items from a cart, purchasing and also subsequence purchasing if a user is still active. For example, one user may visit this website, view a number of mobile phone, add one to the cart and purchasing a mobile phone. Then, the user may continue to view phone accessory and purchase a head phone after that.

As far as I am aware of, this datasets hasn’t been studied by any recommendation papers despite its real world characteristic and highness in quality; Kaggle rates this dataset usability score at 10.0. (It is used in a [Transformers4Rec SBRS Tutorial](https://nvidia-merlin.github.io/Transformers4Rec/main/examples/tutorial/index.html).)

*c. A work-breakdown structure for the individual tasks with time estimates (hours or days) for dataset collection; designing and building an appropriate network; training and fine-tuning that network; building an application to present the results; writing the final report; preparing the presentation of your work.*

- (10 Hours) Clean up, structure and establish workflow to train and test the model on the chosen dataset.
- (10 Hours) Review & Implement generic deep learning methods as baselines ([RNN]).
- (15 Hours) Implementing the state-of-the-art architecture (To be determined from [GNN], [Transformers4Rec], [SGNN-HN] and [TAGNN]).
- (15 Hours) Optimizing promising models.
- (10 Hours) Develop a demo application.
- (5 Hours) Write the final paper & Prepare the presentation.
