import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. creating ds
# Sample data: hours studied and pass/fail outcome
data = {
    'hours_studied': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'pass': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# 2. prep data
# Features and target variable
X = df[['hours_studied']]
y = df['pass']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 3. train the model
# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# 4. Make predictions
y_pred = model.predict(X_test)


# 5. Evaluate model

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)


