import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the email data
email_data = pd.read_csv("email.csv")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(email_data["Text"], email_data["Class"], test_size=0.2)

# Convert the email text into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = naive_bayes.score(X_test, y_test)
print("Accuracy:", accuracy)

# Predict the class label of a new email
new_email = "Hey Billy! Enter our sweepstakes for a chance to win!"
new_email = vectorizer.transform([new_email])
prediction = naive_bayes.predict(new_email)


if prediction[0] == 'not spam':
    print(f"This email is not spam: {new_email}")
else:
    print(f"This email is spam: {new_email}")

# Predict the class label of a new email
new_email = "Hey Billy! What time is our next meeting?"
new_email = vectorizer.transform([new_email])
prediction = naive_bayes.predict(new_email)


if prediction[0] == 'not spam':
    print(f"This email is not spam: {new_email}")
else:
    print(f"This email is spam: {new_email}")