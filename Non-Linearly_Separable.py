import numpy as np
import matplotlib.pyplot as plt

# f(x) = xw1 + yw2 + xyw3 + (x)^2 * w4 + b
X_train = [[1, 1], [9.4, 6.4], [2.5, 2.1], [8, 7.7], [0.5, 2.2],
           [7.9, 8.4], [7, 7], [2.8, 0.8], [1.2, 3], [7.8, 6.1]]

# feature engineering (adding two new features) to fit the model
for i in range(len(X_train)):
    X_train[i].append(X_train[i][0] * X_train[i][1])
    X_train[i].append(X_train[i][0] ** 2)

X_train = np.array(X_train)

Y_train = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0]

# m is the number of training examples and n is the number of input features
m, n = X_train.shape


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
w_in = np.array([0, 0, 0, 0])
b_in = 0
# alpha is the learning rate
alpha = 0.003
iters = 10000
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


def plot_decision_boundary(X_train, y_train, w_out, b_out):
    def equation(x, y):
        return w_out[0] * x + w_out[1] * y + w_out[2] * x * y + w_out[3] * (x ** 2) + b_out

    x = [point[0] for point in X_train]
    y = [point[1] for point in X_train]
    fig, ax = plt.subplots()

    for i, t in enumerate(y_train):
        if t == 1:
            ax.scatter(x[i], y[i], marker='x', color='red')
        else:
            ax.scatter(x[i], y[i], marker='o', color='blue')

    # plot decision boundary
    # Define the range of x and y values
    x_values = np.linspace(0, 10, 100)
    y_values = np.linspace(0, 10, 100)

    # Create a meshgrid from the x and y values
    X, Y = np.meshgrid(x_values, y_values)

    # Evaluate the equation on the meshgrid
    Z = equation(X, Y)

    # Plot the contour of the equation
    plt.contour(X, Y, Z, levels=[0], colors='black')

    # Label the plot
    plt.title('4.16x - 9.53y - 13.05x^2 + 16.51xy - 0.7 = 0')
    plt.xlabel('x')
    plt.ylabel('y')

    # Show the plot
    plt.show()


plot_decision_boundary(X_train, Y_train, w_trained, b_trained)
