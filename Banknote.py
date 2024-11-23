"""
Program Name:  Banknote Authentication Analysis
Description:   This program fetches the Banknote Authentication dataset, then trains a Random Forest Classiefier 
               to classify the notes as either real (1) or fake (0).
              
Date Created:  24-11-2024
Last Modified: 24-11-2024
Version:       1.0
Language:      Python

How to use:
    python banknote_authentication.py

Dependencies:
    - pandas
    - requests
    - os
    - ucimlrepo
    - scikit-learn

Notes:
    - The dataset is from the UCI Machine Learning Repository. 
    - Don't forget to pip install dependencies!
"""


import pandas as pd
import requests
import os
from io import StringIO
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# URL of the data 
url = "https://archive.ics.uci.edu/dataset/267/banknote+authentication"


# Get dataset using ucirepo
banknote_authentication = fetch_ucirepo(id=267)

# data (pandas dataframe)
X = banknote_authentication.data.features
y = banknote_authentication.data.targets

# # metadata
# print(banknote_authentication.metadata)

# # variable information
# print(banknote_authentication.variables)

# print("Features (X):", X.head())
# print("Target (y):", y.head())

# X is the features of note in dataframe to which we later assign columns
df_features = pd.DataFrame(X)

# Y is the class fo the note
df_target = pd.DataFrame(y)

#columns added for
df_features.columns = ['variance', 'skewness', 'curtosis', 'entropy']
df_target.columns = ['class']

df = pd.concat([df_features, df_target], axis=1)

# print(df.head())
# print(df.describe())
# print(df.info())

""" 
First we run a simple random forest algorithm just to see what this yields
"""

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluations
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate False Positive Rate, this indicates how many fake banknotes pass as real. Lower = better."
FP = cm[1, 0]
TN = cm[1, 1]
FPR = FP / (FP + TN)

# Output the result
print(f"False Positive Rate (FPR): {FPR:.2f}")


""" My results, 

Accuracy: 0.9927272727272727
Classification Report:
               precision    recall  f1-score   support

           0       0.99      1.00      0.99       148
           1       1.00      0.98      0.99       127

    accuracy                           0.99       275
   macro avg       0.99      0.99      0.99       275
weighted avg       0.99      0.99      0.99       275

And most importantly FPR: False Positive Rate (FPR): 0.02
"""