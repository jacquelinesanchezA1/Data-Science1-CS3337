from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encoding categorical variables
df['Outlook'] = df['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rain': 2})
df['Temperature'] = df['Temperature'].map({'Hot': 0, 'Mild': 1, 'Cool': 2})
df['Humidity'] = df['Humidity'].map({'High': 0, 'Normal': 1})
df['Play Tennis'] = df['Play Tennis'].map({'No': 0, 'Yes': 1})

# Features and target
X = df[['Outlook', 'Temperature', 'Humidity']]
y = df['Play Tennis']

# Train the model with Gini index
clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(X, y)

# Visualize the tree
fig, ax = plt.subplots(figsize=(12, 8))
tree.plot_tree(clf, feature_names=['Outlook', 'Temperature', 'Humidity'], class_names=['No', 'Yes'], filled=True, ax=ax)
plt.show()
