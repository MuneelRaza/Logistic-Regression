import numpy as np
import matplotlib.pyplot as plt

# f(x) = xw1 + yw2 + b
X_train = np.array([[1, 1], [9.4, 6.4], [2.5, 2.1], [8, 7.7], [0.5, 2.2],
                    [7.9, 8.4], [7, 7], [2.8, 0.8], [1.2, 3], [7.8, 6.1]])

Y_train = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 1])

# m is the number of training examples and n is the number of input features
m, n = X_train.shape

# plot data
for i in range(m):
    if Y_train[i] == 1:
        plt.scatter(X_train[i][0], X_train[i][1], c="red")
    else:
        plt.scatter(X_train[i][0], X_train[i][1], c="blue", marker="x")

plt.xlim(-4, 10)
plt.ylim(-4, 10)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linearly Separable Data")
plt.show()


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def compute_cost(X, y, w, b):
    total_cost = 0
    for i in range(m):
        # computing models prediction on given parameters (w, b)
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        # computing the loss on the ith training example
        loss = -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
        # adding the squared error in total cost
        total_cost = total_cost + loss
    # computing the actual cost on these parameters
    total_cost = total_cost / m

    return total_cost


def compute_gradient(X, y, w, b):
    dj_db = 0
    dj_dw = np.zeros(n)

    for i in range(m):
        # computing models prediction on given parameters (w, b)
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        # computing the error (difference between prediction and actual true label)
        err = (f_wb - y[i])
        # computing gradient for each feature j
        for j in range(n):
            dj_dw[j] = dj_dw[j] + (err * X[i, j])
        dj_db = dj_db + err

    dj_db = dj_db / m
    dj_dw = dj_dw / m
    return dj_db, dj_dw


def gradient_descent(X, y, w, b, alpha, iters):
    # creating an array to save the cost history for plotting
    cost_history = np.zeros(iters)
    # a simple array that stores indexes of iterations
    iterations = np.arange(iters)
    for i in range(iters):
        cost_history[i] = compute_cost(X, y, w, b)
        dj_db, dj_dw = compute_gradient(X, y, w, b)
        # simultaneously update the parameters w and b
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

    return w, b, cost_history, iterations


def predict_training_loss(X, w, b):
    prediction = np.zeros(m)
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        if f_wb >= 0.5:
            prediction[i] = 1
        else:
            prediction[i] = 0

    return prediction


# initializing parameters
w_in = np.array([-0.6, 0.75])
b_in = 0.5
# alpha is the learning rate
alpha = 0.02
iters = 250
# training the model
w_trained, b_trained, cost_history, iterations = gradient_descent(X_train, Y_train, w_in, b_in, alpha, iters)

print("Trained weights", w_trained)
print("Trained bias", b_trained)

predictions = predict_training_loss(X_train, w_trained, b_trained)

print("Actual labels", Y_train)
print("Predicted labels", predictions)

# # plot cost history vs iterations

plt.plot(iterations, cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Iterations vs Cost")
plt.show()

# plot the decision boundary
for i in range(m):
    if Y_train[i] == 1:
        plt.scatter(X_train[i][0], X_train[i][1], c="red")
    else:
        plt.scatter(X_train[i][0], X_train[i][1], c="blue", marker="x")

plt.xlim(-4, 10)
plt.ylim(-4, 10)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linearly Separable Data")
x = np.linspace(-10, 10, 100)
y = -(w_trained[0] * x + b_trained) / w_trained[1]
plt.plot(x, y)
plt.show()

