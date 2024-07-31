import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Creating the dataset
data = {
    'Hair': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Gives_Birth': ['Yes', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Warm-blooded': ['Yes', 'Yes', 'Yes', 'No', 'No', 'Yes'],
    'Mammal': ['Yes', 'No', 'Yes', 'No', 'No', 'No']
}

df = pd.DataFrame(data)

# Encoding the categorical features to numerical values
label_encoders = {}
for column in df.columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Splitting features and labels
X = df[['Hair', 'Gives_Birth', 'Warm-blooded']]
y = df['Mammal']  # target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training the Naive Bayes Model
model = CategoricalNB()
model.fit(X_train, y_train)

# Predicting the test set results
y_pred = model.predict(X_test)

# Evaluating the model on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Example animal to classify
new_animal = pd.DataFrame({
    'Hair': ['Yes'],
    'Gives_Birth': ['Yes'],
    'Warm-blooded': ['No']
})

# Encoding the new animal's features
for column in new_animal.columns:
    new_animal[column] = label_encoders[column].transform(new_animal[column])

# Predicting whether the new animal is a mammal
prediction = model.predict(new_animal)
predicted_label = label_encoders['Mammal'].inverse_transform(prediction)

print(f'The new animal is classified as: {predicted_label[0]}')
