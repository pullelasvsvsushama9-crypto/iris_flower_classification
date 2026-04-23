from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pickle

# load dataset
iris = load_iris()

X = iris.data
y = iris.target

# train model
model = KNeighborsClassifier()
model.fit(X, y)

# save model
pickle.dump(model, open("iris_model.pkl", "wb"))

print("Model trained successfully!")