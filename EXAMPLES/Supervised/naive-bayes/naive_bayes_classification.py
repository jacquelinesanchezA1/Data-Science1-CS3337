import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

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

# Creating and training the Naive Bayes Model
model = CategoricalNB()
model.fit(X, y)

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
