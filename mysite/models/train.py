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
