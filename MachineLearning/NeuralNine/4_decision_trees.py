from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()

X = data.data
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

clf2 = RandomForestClassifier()
clf2.fit(x_train, y_train)

print("Decision Tree: ", clf.score(x_test, y_test))
print("Random Forest: ", clf2.score(x_test, y_test))
