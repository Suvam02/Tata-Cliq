import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
data = pd.read_csv('train.csv')
X = data.iloc[:,2].values
y = data.iloc[:, -2].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)
# Split the data into training and testing sets
## X_train, X_test, y_train, y_test = train_test_split(data['City'], data['Gender'], test_size=0.2, random_state=42)

# Convert the city names to feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier on the training data
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Evaluate the classifier on the testing data
accuracy = clf.score(X_test_vec, y_test)
print(f'Accuracy: {accuracy}')

# Use the classifier to predict the gender of people living in new cities

new_test=pd.read_csv('test.csv')['p_brand']
new_test_vec = vectorizer.transform(new_test)
predictions = clf.predict(new_test_vec)
print(f'Predictions: {predictions}')
