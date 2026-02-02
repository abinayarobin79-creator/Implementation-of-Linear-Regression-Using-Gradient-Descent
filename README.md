# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe. 
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Abinaya R
RegisterNumber:  212225230004
*/
import numpy as np
import matplotlib.pyplot as plt

# Sample training data (Population of City, Profit)
X = np.array([6.1101, 5.5277, 8.5186, 7.0032, 5.8598, 8.3829])
y = np.array([17.592, 9.1302, 13.662, 11.854, 6.8233, 13.662])

# Number of samples
m = len(y)

# Add column of 1s for bias (intercept) term
X_b = np.c_[np.ones((m, 1)), X]

# Initialize parameters (theta0, theta1)
theta = np.zeros(2)

# Gradient Descent settings
alpha = 0.01
iterations = 1500

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Gradient Descent algorithm
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        gradient = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta = theta - alpha * gradient
    return theta

# Train the model
theta = gradient_descent(X_b, y, theta, alpha, iterations)
print("Theta values:", theta)

# Predict profit for a given city population
population = float(input("Enter city population: "))
prediction = theta[0] + theta[1] * population
print("Predicted Profit:", prediction)

# Plot the data and regression line
plt.scatter(X, y, label="Training Data")
plt.plot(X, X_b.dot(theta), label="Regression Line")
plt.xlabel("Population of City")
plt.ylabel("Profit")
plt.legend()
plt.show()

```

## Output:
![linear regression using gradient descent](sam.png)

<img width="442" height="89" alt="Screenshot 2026-02-02 111848" src="https://github.com/user-attachments/assets/4d2ef814-21d6-476b-8f2a-f8a9b9e7f5b2" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
