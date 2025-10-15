# Iris-Flower-Classification
# Iris Flower Classification Project
# Author: Abhishek Kumar Rajak
# Description: Predict the species of Iris flowers using KNN and Decision Tree classifiers

# 📦 Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📥 Step 2: Load Dataset
df = sns.load_dataset('iris')
print("Dataset Preview:")
print(df.head())

# 🔍 Step 3: Explore Data
print("\nClass Distribution:")
print(df['species'].value_counts())
sns.pairplot(df, hue='species')
plt.suptitle("Iris Feature Distribution", y=1.02)
plt.show()

# 🎯 Step 4: Prepare Data
X = df.drop('species', axis=1)
y = df['species']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 📊 Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 🤖 Step 6: Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)

# 🌳 Step 7: Train Decision Tree Classifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)

# 🧪 Step 8: Evaluate Models
print("\n📈 KNN Classifier Evaluation:")
print("Accuracy:", accuracy_score(y_test, knn_preds))
print(classification_report(y_test, knn_preds, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_preds))

print("\n🌲 Decision Tree Classifier Evaluation:")
print("Accuracy:", accuracy_score(y_test, tree_preds))
print(classification_report(y_test, tree_preds, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, tree_preds))
