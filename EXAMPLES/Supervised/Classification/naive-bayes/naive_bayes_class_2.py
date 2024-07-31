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

# Creating and training the Naive Bayes model
model = CategoricalNB()
model.fit(X, y)

# Example animals to classify
new_animals = pd.DataFrame({
    'Hair': ['Yes', 'Yes'],
    'Gives_Birth': ['Yes', 'Yes'],
    'Warm-blooded': ['No', 'Yes']
})

# Encoding the new animals' features
for column in new_animals.columns:
    new_animals[column] = label_encoders[column].transform(new_animals[column])

# Predicting whether the new animals are mammals
predictions = model.predict(new_animals)
predicted_labels = label_encoders['Mammal'].inverse_transform(predictions)

# Printing the classifications
for i, animal in new_animals.iterrows():
    print(f'New animal {i+1} with features {animal.to_dict()} is classified as: {predicted_labels[i]}')
