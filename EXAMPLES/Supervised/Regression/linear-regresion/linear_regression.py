import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# read the dataset
housing_data = pd.read_csv('USA_Housing.csv')
housing_data.head()

y = housing_data['Price']
X = housing_data.drop(columns=['Price', 'Address'])

# split the data to training and test portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# apply the model to the training set
lm = LinearRegression()
lm.fit(X_train, y_train)

# Model Evaluation
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.show()

# since we are dealing with a linear model it's important to view the residuals
residuals = y_test-predictions
sns.histplot(residuals)
plt.show()

# lets evaluate the model
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
