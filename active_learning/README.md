## Active Learning

Active learning is a type of machine learning where the algorithm actively selects the examples it wants to be labeled by a human annotator. The goal of active learning is to optimize the use of human annotation effort by actively choosing the examples that are most informative for learning.

For example, an active learning algorithm could be used for text classification where the algorithm selects the examples it wants to be labeled by a human annotator. The algorithm uses its current model to select the examples that it is most uncertain about, and a human annotator provides the labels for these examples. The labeled examples are then used to train the model, which can then be used to make predictions on the remaining unlabeled examples. This process can be repeated iteratively, with the algorithm selecting the most uncertain examples for labeling each time.

Active learning is often used when labeled data is scarce or expensive to obtain, and can lead to improved performance compared to supervised learning with limited labeled data. The success of active learning depends on the choice of query strategy and the quality of the annotator. In this chapter, we will discuss the two active learning techniques listed below:

    -   Query-by-committee
    -   Uncertainty sampling