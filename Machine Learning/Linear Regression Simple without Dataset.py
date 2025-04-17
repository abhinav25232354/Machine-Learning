from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = [[1], [2], [3], [4], [5]]  # Features
y = [2, 4, 5, 4, 5]  # Target

model = LinearRegression()  
model.fit(X, y)  # Train the model
y_pred = model.predict(X)  # Predict values

print("Slope:", model.coef_[0])  
print("Intercept:", model.intercept_)  
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.show()