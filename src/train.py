from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib 


kneighbors = KNeighborsClassifier(n_neighbors=3)

iris = load_iris()
x = iris.data # shape (150,4)
y = iris.target # shape (150,)
print(iris.feature_names, iris.target_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("predictions:", y_pred[:5])
print("actual labels:", y_test[:5]

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


model2 = KNeighborsClassifier (n_neighbors=5)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)
y_pred3 = model3.predict(X_test)
print ("example of overfitting", accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Setup Sample Data (Actual values vs Predicted values)
# 0 = Negative Class, 1 = Positive Class
actual    = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
predicted = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]

# 2. Compute the confusion matrix
cm = confusion_matrix(actual, predicted)

# 3. Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Example: Train a simple model
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
model = LogisticRegression()
model.fit(X, y)

# --- Save the model ---
# The typical file extension is '.joblib' or '.pkl' or '.gz' if compressed
filename = 'my_trained_model.joblib'
joblib.dump(model, filename)

print(f"Model saved to {filename}")

import joblib

# --- Load the model ---
filename = 'my_trained_model.joblib'
loaded_model = joblib.load(filename)

print(f"Model loaded from {filename}")

# Now you can use the loaded model, for example, to make predictions
# Assuming 'new_data' is compatible input for the model
# predictions = loaded_model.predict(new_data)
# print(predictions)

