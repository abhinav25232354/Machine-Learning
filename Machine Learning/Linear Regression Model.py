import matplotlib.pyplot as plt # pip install matplotlib
import numpy as np # pip install numpy
# pip install scikit-learn for sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
# scikit-learn is a machine learning ibrary which includes 

# Diabetes check linear regression model
diabetes = datasets.load_diabetes()
# diabetes.keys()

# Taking only some features
# Can achieve for whole features in the data set by removing [:, np.newaxis, 2] slicing. 
diabetes_x = diabetes.data[:, np.newaxis, 2]
# print(diabetes_x)
diabetes_x_train = diabetes_x[:-30] # Taking first 30 for training
diabetes_x_test = diabetes_x[-30:] # keeping last 30 for testing model dependencies

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)
diabetes_y_predict = model.predict(diabetes_x_test)

print(f"Mean Squared Error: {mean_squared_error(diabetes_y_test, diabetes_y_predict)}")
print(f"Weights: {model.coef_}")
print(f"Intercept: {model.intercept_}")

plt.scatter(diabetes_x_test, diabetes_y_test)
plt.plot(diabetes_x_test, diabetes_y_predict)

plt.show()