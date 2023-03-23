# Examples

This folder contains working code snippets for various machine learning projects.

## Getting Started

1. Clone this repository to your local machine.
2. Navigate to the `Examples` folder.
3. Run the desired Python file to see the example in action.

## Example 1: Spam Email Detector

The `spam_email.py` file contains code for a Naive Bayes classifier that can sort spam and not spam email data.

### Prerequisites

- Python 3.x
- pandas
- scikit-learn

### Usage

1. Place your email data in a CSV file called `email.csv` with the following format:

```csv
Text,Class
"Dear Friend, Congratulations! You have won a free gift. Claim it now!",spam
"Hello, just checking in. How are you doing today?",not spam
"Get rich quick! Invest now and watch your fortune grow!",spam
"Hi there, this is a friendly reminder about your meeting tomorrow.",not spam
```

2. Run `spam_email.py` to train the classifier and see it in action.

### Code Explanation

1. Load the email data using `pd.read_csv`.
2. Split the data into training and test sets using `train_test_split`.
3. Convert the email text into numerical features using `CountVectorizer`.
4. Train the Naive Bayes classifier using `MultinomialNB`.
5. Evaluate the classifier on the test data using the `score` method.
6. Predict the class label of a new email using the `predict` method and check if it is spam or not.

### Results

The output of `spam_email.py` will be the accuracy of the classifier on the test data, as well as the predicted class label of two new emails.

```
Accuracy: 1.0
This email is spam:   (0, 16)	1
  (0, 57)	1
This email is not spam:   (0, 29)	1
  (0, 33)	1
  (0, 35)	1
```

