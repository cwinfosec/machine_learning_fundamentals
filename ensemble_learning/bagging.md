### Bootstrap Aggregating

Ensemble learning is a type of machine learning technique that combines multiple models to produce a more accurate and reliable prediction than any single model could. Boostrap aggregating, also known as bagging, is a technique used in ensemble learning to reduce the variance of a single model and improve the accuracy of the model’s predictions. It works by creating multiple versions of the same model, each with different randomly selected subsets of the training data. The predictions from each model are then averaged to create a final prediction.

Boostrap aggregating is especially useful when training models with very high variance, such as decision trees. By creating multiple versions of the model and averaging their predictions, the variance of the model is reduced and the accuracy of the predictions is improved.

Bootstrap aggregating (also known as bagging) is an ensemble learning technique that combines multiple models to create a more powerful and robust model. It works by taking multiple samples from a dataset with replacement, and training a model on each of the samples. The models are then combined to form a single model, which is more accurate and robust than any of the individual models.

The math behind bootstrap aggregating is based on the probability theory of sampling with replacement. This states that the probability of sampling any particular observation from a dataset is equal to the probability of sampling any other observation. This means that the probability of a particular observation being selected is independent of the other observations in the dataset. 

Mathematically, this can be expressed as:
```
P(x_i) = P(x_j) 
```
Where `x_i` and `x_j` are any two observations in the dataset. 

This means that when sampling with replacement, the probability of any particular observation being selected is equal to the probability of any other observation being selected. This is what makes bootstrap aggregating possible.

A simple use case example of boostrap aggregating would be a model that predicts stock prices. By creating multiple versions of the model and averaging their predictions, the variance of the model is reduced and the accuracy of the predictions is improved. This can be useful in helping investors decide when to buy and sell stocks.

Here's an example of using bootstrapped ensembling for stock price prediction in Python:
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the stock data
df = pd.read_csv("stock_prices.csv")

# Split the data into features and labels
X = df.drop("Close", axis=1)
y = df["Close"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create an array to store the predictions from each model
predictions = np.zeros(X_test.shape[0])

# Train and predict with 100 decision tree models
for i in range(100):
    # Bootstrap the data by randomly selecting samples with replacement
    bootstrapped_indices = np.random.choice(range(X_train.shape[0]), size=X_train.shape[0], replace=True)
    X_train_bootstrapped = X_train.iloc[bootstrapped_indices]
    y_train_bootstrapped = y_train.iloc[bootstrapped_indices]
    
    # Train the decision tree model
    model = RandomForestRegressor()
    model.fit(X_train_bootstrapped, y_train_bootstrapped)
    
    # Predict with the model
    predictions += model.predict(X_test)

# Average the predictions from all models
predictions /= 100
```

This code loads the stock data from a CSV file stock_prices.csv into a Pandas DataFrame df. It then splits the data into features and labels, with the features stored in X and the labels stored in y. The data is then split into a training set and a testing set, with 80% of the data used for training and 20% used for testing.

The code then trains 100 decision tree models using bootstrapped data, with the bootstrapped data generated by randomly selecting samples from the training data with replacement. The predictions from each model are added to the predictions array, and the final prediction for each test sample is obtained by averaging the predictions from all models.

This example demonstrates how bootstrapping can be used to create an ensemble of decision tree models for stock price prediction, with the hope of reducing the variance and improving the accuracy of the predictions.

Here's an example of a `stock_prices.csv` file that could be used with the code for bootstrap aggregating ensemble learning:
```
Date,Open,High,Low,Close,Adj Close,Volume
2020-01-02,148.25,150.00,145.00,147.10,147.10,40355500
2020-01-03,147.00,148.23,145.50,147.23,147.23,33405500
2020-01-06,146.00,148.50,145.00,148.00,148.00,28705500
2020-01-07,148.50,150.00,147.50,149.50,149.50,25505500
2020-01-08,149.50,150.00,148.00,149.50,149.50,22605500
2020-01-09,149.50,151.50,149.00,150.00,150.00,32005500
...
```

This file contains the daily opening, high, low, close, adjusted close, and volume prices for a stock from January 2nd, 2020 to the end of the available data. The code would use this file to train multiple versions of a stock price prediction model, and average their predictions to get a more accurate prediction.