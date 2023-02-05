### Feature extraction

Feature extraction in transfer learning is a process of extracting features from a pre-trained model and using them in a new task. It is a type of knowledge transfer from the pre-trained model to the new task and is used to improve the accuracy of the new task.

In feature extraction, the pre-trained model is used to extract features from the data that are relevant to the new task. The extracted features are then used to train the new model. This approach is useful in situations where there is limited data available to train a new model from scratch. By using the pre-trained model to extract features, the new model can be trained with much less data.

Feature extraction transfer learning is a method of machine learning where a pre-trained model is used to extract features from a given dataset. This method is used to improve the accuracy of a machine learning model by using the pre-trained model as a starting point, rather than training the model from scratch. 

The mathematical equation behind feature extraction transfer learning is as follows: 

Let `D` be the dataset, `M` be the pre-trained model, `F` be the feature extraction function, and `T` be the target model. 

The feature extraction transfer learning equation is: 
```
T = F(M(D))
```
This equation can be read as: The target model `(T)` is equal to the feature extraction function `(F)` applied to the pre-trained model `(M)` applied to the dataset `(D)`.

For instance, a computer vision application may require a large dataset to train a model from scratch. However, by using feature extraction, the model can be trained with a much smaller dataset. The pre-trained model is used to extract features from the images in the dataset and then the extracted features are used to train the new model. This approach can significantly reduce the amount of data needed to train the model and improve its accuracy.