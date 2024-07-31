import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
housing_data = pd.read_csv('USA_Housing.csv')

# Inspect data types
print(housing_data.dtypes)

# Convert data types if necessary (example: converting 'object' types to 'float')
# Here assuming all columns except 'Address' (if it exists) should be numeric
for column in housing_data.columns:
    if housing_data[column].dtype == 'object' and column != 'Address':
        housing_data[column] = pd.to_numeric(housing_data[column], errors='coerce')

# Check for any remaining non-numeric data
print(housing_data.dtypes)

# Define the target variable (y) and features (X)
y = housing_data['Price']
X = housing_data.drop(columns=['Price', 'Address'])

# Split the data into training and test portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Apply the model to the training set
lm = LinearRegression()
lm.fit(X_train, y_train)

# Model Evaluation
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

# Since we are dealing with a linear model, it's important to view the residuals
residuals = y_test - predictions
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.show()

# Let's evaluate the model
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
