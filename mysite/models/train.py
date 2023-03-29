from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# save trained model
joblib.dump(model, 'iris_model.pkl')

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# load iris dataset from scikit-learn
from sklearn.datasets import load_iris
iris = load_iris()

# create Pandas dataframe from iris dataset
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], df['target'], test_size=0.2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# save model to disk
joblib.dump(model, 'iris_model.pkl')
"""