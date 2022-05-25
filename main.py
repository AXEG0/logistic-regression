import math
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class SimpleLogisticRegression:

    def __init__(self, l_rate=0.01, max_iter=1000):
        self.l_rate = l_rate
        self.max_iter = max_iter

    def sigmoid(self, t):
        return 1 / (1 + math.exp(-t))

    def predict_proba(self, row, coef_):
        t = sum(np.insert(row.values, 0, 1) * coef_)
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        self.coef_ = np.zeros(len(X_train.columns) + 1)
        for _ in range(self.max_iter):
            for index, row in X_train.iterrows():
                y_hat = self.predict_proba(row, self.coef_)
                calc_term = (y_hat - y_train[index]) * y_hat * (1 - y_hat)
                self.coef_ -= self.l_rate * calc_term * np.insert(row.values, 0, 1)
        return self.coef_

    def predict(self, X_test, cut_off=0.5):
        predictions = list()
        for i, row in X_test.iterrows():
            y_hat = self.predict_proba(row, self.coef_)
            if y_hat < cut_off:
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions


# load dataframe
df = load_breast_cancer(as_frame=True)
# define features columns
columns = ["worst concave points", "worst perimeter", "worst radius"]
# define features and target
X, y = df.data[columns], df.target
# standardize features by removing the mean and scaling to unit variance
X = StandardScaler().fit_transform(X)
# covert data to dataframe from numpy array
X = pd.DataFrame(X)
# split data to train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
# create model objects
simple_model = SimpleLogisticRegression(l_rate=0.01, max_iter=1000)
sklearn_model = LogisticRegression(max_iter=1000)
# get the coefficients of models
simple_coef_ = simple_model.fit_mse(X_train, y_train)
sklearn_coef_ = sklearn_model.fit(X_train, y_train).coef_
# get predicted values with the test set
simple_y_pred = simple_model.predict(X_test, cut_off=0.5)
sklearn_y_pred = sklearn_model.predict(X_test)
# get the accuracy
simple_accuracy = accuracy_score(y_test, simple_y_pred)
sklearn_accuracy = accuracy_score(y_test, sklearn_y_pred)
# show result
print(f"""Simple Logistic Regression model result:
Coefficients: {simple_coef_.tolist()}
Accuracy: {simple_accuracy}\n
sklearn Logistic Regression model result:
Coefficients: {sklearn_model.intercept_.tolist() + sklearn_coef_[0].tolist()}
Accuracy: {sklearn_accuracy}\n
Equality of accuracy: {simple_accuracy == sklearn_accuracy}
""")

# Simple Logistic Regression model result:
# Coefficients: [1.2657813061579848, -2.921414880939942, -2.0078141181387705, -3.0120370953096014]
# Accuracy: 0.9473684210526315
#
# sklearn Logistic Regression model result:
# Coefficients: [1.102288326658557, -2.5095945427704947, -1.8057746081832773, -2.235459634254033]
# Accuracy: 0.9473684210526315
#
# Equality of accuracy: True


