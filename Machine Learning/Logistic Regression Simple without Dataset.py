import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

model = LogisticRegression()
model.fit(X, y)
y_prob = model.predict_proba(X)[:, 1]  # Probability of class 1

plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, y_prob, color='red', label="Sigmoid Curve")
plt.legend()
plt.show()