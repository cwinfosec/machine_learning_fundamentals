## Transfer Learning

Transfer learning is a type of machine learning where a model trained on one task is used as the starting point for a model on a related but different task. The goal of transfer learning is to leverage the knowledge gained from solving one problem to improve the performance on a different but related problem.

For example, a transfer learning algorithm could be used for image classification where a model trained on a large dataset of natural images is used as a starting point for a model on a smaller dataset of medical images. The idea is that the model has already learned useful features and representations of natural images, which can be transferred to the task of classifying medical images. This can lead to improved performance compared to training a model from scratch on the smaller dataset of medical images.

Transfer learning is often used when a large amount of labeled data is not available for a particular task, and can lead to improved performance compared to training a model from scratch on limited data. The success of transfer learning depends on the relatedness of the source and target tasks, as well as the choice of model and the adaptation methods used. In this chapter, we will discuss the two transfer learning techniques listed below:

    -   Fine-tuning
    -   Feature extraction