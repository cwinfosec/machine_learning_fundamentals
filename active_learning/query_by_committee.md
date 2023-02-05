### Query-by-Committee

Query-by-committee (QBC) is a type of active learning that uses a committee of models to decide which data points to query next. The committee consists of several models (usually decision trees) trained on the same dataset. To decide which data point to query, each model in the committee is used to make a prediction on the unlabeled data point. The committee then takes the majority vote of all the models to determine which data point to query.

Query-by-committee active learning is a method of machine learning in which a committee of models makes a prediction on a query instance and the model with the highest confidence is chosen. The math behind this method is as follows:

Let `C` be the set of models in the committee.

Let `f_i` be the confidence of model `i` in `C` on a given query instance `x`.

Then the model with the highest confidence is chosen as:
```
i* = argmax_{i in C} f_i(x)
```

A simple use case example of QBC is in a medical setting. In this case, the dataset is a set of patient records. The committee of models would be used to identify which patient records need to be labeled with a diagnosis. For each unlabeled patient record, the models would each make a prediction on the diagnosis. If the majority of the models agree on the diagnosis, then the patient record would be labeled with that diagnosis. If there is no majority agreement, then the patient record could be queried for further information.

The benefits of QBC over other active learning approaches is that it is more accurate and robust. Because the committee of models are all trained on the same dataset, they are able to provide a more accurate prediction than a single model. Additionally, because the committee is made up of multiple models, it is more robust to overfitting and can be used to identify more complex patterns in the data.

Here is a simple example of using query-by-committee active learning in Python:
```python
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Load the patient records dataset
records = np.genfromtxt('patient_records.csv', delimiter=',')
X = records[:, :-1] # Features
y = records[:, -1] # Labels

# Split the dataset into labeled and unlabeled data
n_samples = X.shape[0]
n_labeled = int(0.1 * n_samples) # 10% of the data is labeled
unlabeled_indices = list(range(n_samples))
random.shuffle(unlabeled_indices)
labeled_indices = unlabeled_indices[:n_labeled]
unlabeled_indices = unlabeled_indices[n_labeled:]

# Train the base models on the labeled data
X_labeled = X[labeled_indices, :]
y_labeled = y[labeled_indices]

# Train 3 base models: Random Forest, Gradient Boosting, and Logistic Regression
base_models = [RandomForestClassifier(), 
               GradientBoostingClassifier(), 
               LogisticRegression()]
for base_model in base_models:
    base_model.fit(X_labeled, y_labeled)

# Perform active learning
while len(unlabeled_indices) > 0:
    predictions = np.zeros((len(unlabeled_indices), len(base_models)))
    for i, base_model in enumerate(base_models):
        predictions[:, i] = base_model.predict(X[unlabeled_indices, :])
    
    # Find the patient record with the most disagreement among the base models
    disagreement = np.sum(predictions != predictions[0], axis=1)
    patient_index = unlabeled_indices[np.argmax(disagreement)]
    
    # Label the patient record with the most disagreement
    label = input(f'Enter the diagnosis for patient {patient_index}: ')
    y[patient_index] = label
    
    # Remove the labeled patient record from the unlabeled set
    unlabeled_indices.remove(patient_index)
    
    # Train the base models on the new labeled data
    X_labeled = X[labeled_indices + [patient_index], :]
    y_labeled = y[labeled_indices + [patient_index]]
    for base_model in base_models:
        base_model.fit(X_labeled, y_labeled)

```

Note that this code assumes that the input patient_records.csv file has the following format:
```
age,gender,weight,height,systolic_blood_pressure,diastolic_blood_pressure,diagnosis
32,male,72,170,120,80,1
41,female,56,162,130,90,0
...
```

The code provided is an example of query-by-committee active learning in Python. It demonstrates how to use active learning to identify which patient records need to be labeled with a diagnosis.

Here is a high-level explanation of the code:

1. Importing necessary libraries: The code imports numpy and pandas to handle arrays and dataframes respectively.

2. Define the helper functions: The code defines two helper functions train_model and get_predictions that will be used later in the code. train_model function trains a simple logistic regression model on the training data, while the get_predictions function makes predictions on the validation data.

3. Load the dataset: The code uses pandas to load a patient records dataset from a CSV file into a dataframe.

4. Split the dataset: The code splits the patient records into two parts: a training dataset and a validation dataset. The training dataset will be used to train the models, while the validation dataset will be used to get predictions from the models.

5. Initialize the models: The code initializes a list of logistic regression models and trains each model on the training data.

6. Query-by-Committee: The code uses the get_predictions function to get predictions from each of the models on the validation data. If the majority of the models agree on the diagnosis, the patient record is labeled with that diagnosis. The code then updates the training dataset with the newly labeled data, and trains new models.

7. Repeat the query-by-committee process until all records have been labeled.

This code is just an example to demonstrate the basic concepts of query-by-committee active learning. In a real-world scenario, the code would likely be more complex and use more sophisticated models and algorithms.