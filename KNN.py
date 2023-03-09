
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


# Load the dataset from the train dataset > Link in PPT Slide 4
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


# Convert the p_brand names to feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a KNN classifier on the training data
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p = 2)
classifier.fit(X_train_vec, y_train)


# Evaluate the classifier on the testing data
accuracy = classifier.score(X_test_vec, y_test)
print(f'Accuracy: {accuracy}')

# Use the classifier to predict the gender of people 
# Load the dataset from the test dataset > Link in PPT Slide 4
new_test=pd.read_csv('test.csv')['p_brand']
new_test_vec = vectorizer.transform(new_test)
predictions = classifier.predict(new_test_vec)
print(f'Predictions: {predictions}')

t=pd.read_csv('Submission_Team Hacksmiths.csv')
t['customer_gender_pred']=predictions
t.to_csv('Submission_Team Hacksmiths.csv')

