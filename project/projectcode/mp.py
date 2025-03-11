# import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np
from scipy import optimize
from functools import partial
import multiprocessing as mp
import timeit
import torch
import torch.nn.functional as F

MAXITER = 600
TRAINING_DIR = Path("./training/")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Create Your Own Artificial Neural Network for Multi-class Classification (With Python)
Philip Mocz (2023), @PMocz

Create and train your own artificial neural network to classify images of galaxies from SDSS/the Galaxy Zoo project.
"""


def g(x):
    """sigmoid function"""
    return 1.0 / (1.0 + np.exp(-x))


def grad_g(x):
    """gradient of sigmoid function"""
    gx = g(x)
    return gx * (1.0 - gx)


def predict(Theta1, Theta2, X):
    """Predict labels in a trained three layer classification network.
    Input:
      Theta1       trained weights applied to 1st layer (hidden_layer_size x input_layer_size+1)
      Theta2       trained weights applied to 2nd layer (num_labels x hidden_layer_size+1)
      X            matrix of training data      (m x input_layer_size)
    Output:
      prediction   label prediction
    """
    m = np.shape(X)[0]
    a1 = np.hstack((np.ones((m, 1)), X))
    a2 = g(a1 @ Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), a2))
    a3 = g(a2 @ Theta2.T)

    prediction = np.argmax(a3, 1).reshape((m, 1))
    return prediction


def reshape(theta, input_layer_size, hidden_layer_size, num_labels):
    """reshape theta into Theta1 and Theta2, the weights of our neural network"""
    ncut = hidden_layer_size * (input_layer_size + 1)
    Theta1 = theta[0:ncut].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = theta[ncut:].reshape(num_labels, hidden_layer_size + 1)
    return Theta1, Theta2


def cost_function(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """Neural net cost function for a three layer classification network.
    Input:
      theta               flattened vector of neural net model parameters
      input_layer_size    size of input layer
      hidden_layer_size   size of hidden layer
      num_labels          number of labels
      X                   matrix of training data
      y                   vector of training labels
      lmbda               regularization term
    Output:
      J                   cost function
    """
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
    m = len(y)

    a1 = np.hstack((np.ones((m, 1)), X))
    a2 = g(a1 @ Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), a2))
    a3 = g(a2 @ Theta2.T)

    # create one-hot encoding for y
    y_mtx = 1.0 * (y == 0)
    for k in range(1, num_labels):
        y_mtx = np.hstack((y_mtx, 1.0 * (y == k)))

    # cost function
    J = np.sum(-y_mtx * np.log(a3) - (1.0 - y_mtx) * np.log(1.0 - a3)) / m
    # add regularization
    J += lmbda / (2.0 * m) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))
    return J


def compute_grad_t(t, Theta1, Theta2, X, y, num_labels):
    """
    Compute the gradient contributions (Delta1 and Delta2) for a single training example.
    """
    input_layer_size = X.shape[1]
    # Forward pass for sample t
    a1 = X[t, :].reshape((input_layer_size, 1))
    a1 = np.vstack((1, a1))  # add bias
    z2 = Theta1 @ a1
    a2 = g(z2)
    a2 = np.vstack((1, a2))  # add bias
    a3 = g(Theta2 @ a2)

    # create one-hot vector for label
    y_k = np.zeros((num_labels, 1))
    y_k[int(y[t, 0])] = 1

    # Compute error for output layer and hidden layer
    delta3 = a3 - y_k
    Delta2_t = delta3 @ a2.T
    delta2 = (Theta2[:, 1:].T @ delta3) * grad_g(z2)
    Delta1_t = delta2 @ a1.T

    return Delta1_t, Delta2_t


def gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    if USE_MP and not USE_GPU:
        return mp_gradient(
            theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda
        )
    elif USE_GPU and not USE_MP:
        return torch_gradient(
            theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda
        )
    else:
        return standard_gradient(
            theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda
        )


def process_chunk(chunk_data):
    """
    Process a chunk of training examples to compute gradient contributions.
    """
    (
        chunk_range,
        Theta1,
        Theta2,
        X,
        y,
        input_layer_size,
        hidden_layer_size,
        num_labels,
    ) = chunk_data
    start, end = chunk_range
    Delta1_chunk = np.zeros((hidden_layer_size, input_layer_size + 1))
    Delta2_chunk = np.zeros((num_labels, hidden_layer_size + 1))

    for t in range(start, end):
        a1 = X[t, :].reshape((input_layer_size, 1))
        a1 = np.vstack((1, a1))  # add bias
        z2 = Theta1 @ a1
        a2 = g(z2)
        a2 = np.vstack((1, a2))  # add bias
        a3 = g(Theta2 @ a2)

        y_k = np.zeros((num_labels, 1))
        y_k[int(y[t, 0])] = 1

        delta3 = a3 - y_k
        Delta2_chunk += delta3 @ a2.T
        delta2 = (Theta2[:, 1:].T @ delta3) * grad_g(z2)
        Delta1_chunk += delta2 @ a1.T

    return Delta1_chunk, Delta2_chunk


def mp_gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """
    Neural net cost function gradient for a three layer classification network.
    This version parallelizes the computation over batches of training examples.
    """
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
    m = len(y)

    # process data in chunks, 2 seems to be the magic number
    num_processes = round(mp.cpu_count() / 2)
    chunk_size = max(1, m // num_processes)
    chunks = [(i, min(i + chunk_size, m)) for i in range(0, m, chunk_size)]

    # data for each chunk
    chunk_data = [
        (
            (start, end),
            Theta1,
            Theta2,
            X,
            y,
            input_layer_size,
            hidden_layer_size,
            num_labels,
        )
        for start, end in chunks
    ]

    # chunks in parallel
    results = pool.map(process_chunk, chunk_data)

    # results
    Delta1 = np.sum([res[0] for res in results], axis=0)
    Delta2 = np.sum([res[1] for res in results], axis=0)

    # gradients
    Theta1_grad = Delta1 / m
    Theta2_grad = Delta2 / m

    Theta1_grad[:, 1:] += (lmbda / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lmbda / m) * Theta2[:, 1:]

    grad_flat = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
    return grad_flat


def standard_gradient(
    theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda
):
    """Neural net cost function gradient for a three layer classification network.
    Input:
      theta               flattened vector of neural net model parameters
      input_layer_size    size of input layer
      hidden_layer_size   size of hidden layer
      num_labels          number of labels
      X                   matrix of training data
      y                   vector of training labels
      lmbda               regularization term
    Output:
      grad                flattened vector of derivatives of the neural network
    """

    # unflatten theta
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)

    # number of training values
    m = len(y)

    # Backpropagation: calculate the gradients Theta1_grad and Theta2_grad:

    Delta1 = np.zeros((hidden_layer_size, input_layer_size + 1))
    Delta2 = np.zeros((num_labels, hidden_layer_size + 1))

    for t in range(m):
        # forward very bad, very slow
        a1 = X[t, :].reshape((input_layer_size, 1))
        a1 = np.vstack((1, a1))  #  +bias
        z2 = Theta1 @ a1
        a2 = g(z2)
        a2 = np.vstack((1, a2))  #  +bias
        a3 = g(Theta2 @ a2)

        # compute error for layer 3
        y_k = np.zeros((num_labels, 1))
        y_k[y[t, 0].astype(int)] = 1
        delta3 = a3 - y_k
        Delta2 += delta3 @ a2.T

        # compute error for layer 2
        delta2 = (Theta2[:, 1:].T @ delta3) * grad_g(z2)
        Delta1 += delta2 @ a1.T

    Theta1_grad = Delta1 / m
    Theta2_grad = Delta2 / m

    # add regularization
    Theta1_grad[:, 1:] += (lmbda / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lmbda / m) * Theta2[:, 1:]

    # flatten gradients
    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return grad


def torch_gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """
    Neural net cost function gradient for a three layer classification network.
    This version uses PyTorch with GPU acceleration when available.
    """

    # convert
    theta_tensor = torch.tensor(
        theta, dtype=torch.float32, device=DEVICE, requires_grad=True
    )
    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    y_tensor = torch.zeros(len(y), num_labels, dtype=torch.float32, device=DEVICE)
    for i in range(len(y)):
        y_tensor[i, int(y[i, 0])] = 1.0

    ncut = hidden_layer_size * (input_layer_size + 1)
    Theta1_tensor = theta_tensor[:ncut].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2_tensor = theta_tensor[ncut:].reshape(num_labels, hidden_layer_size + 1)

    m = X.shape[0]

    a1 = torch.cat([torch.ones(m, 1, device=DEVICE), X_tensor], dim=1)

    # Hidden layer activation
    z2 = torch.mm(a1, Theta1_tensor.t())
    a2 = torch.sigmoid(z2)

    # Add bias to hidden layer
    a2 = torch.cat([torch.ones(m, 1, device=DEVICE), a2], dim=1)

    # Output layer activation
    z3 = torch.mm(a2, Theta2_tensor.t())
    a3 = torch.sigmoid(z3)

    # Cost calculation
    cost = (
        -torch.sum(y_tensor * torch.log(a3) + (1.0 - y_tensor) * torch.log(1.0 - a3))
        / m
    )

    # Add regularization
    reg_term = (lmbda / (2.0 * m)) * (
        torch.sum(Theta1_tensor[:, 1:] ** 2) + torch.sum(Theta2_tensor[:, 1:] ** 2)
    )
    cost += reg_term

    # Compute gradients via autograd
    cost.backward()

    # Return gradients as numpy array
    grad_flat = theta_tensor.grad.cpu().numpy()
    return grad_flat


N_iter = 1
J_min = np.inf
theta_best = []
Js_train = np.array([])
Js_test = np.array([])


def callbackF(
    input_layer_size,
    hidden_layer_size,
    num_labels,
    X,
    y,
    lmbda,
    test,
    test_label,
    theta_k,
):
    """Calculate some stats per iteration and update plot"""
    global N_iter, J_min, theta_best, Js_train, Js_test
    Theta1, Theta2 = reshape(theta_k, input_layer_size, hidden_layer_size, num_labels)
    J = cost_function(
        theta_k, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda
    )
    y_pred = predict(Theta1, Theta2, X)
    accuracy = np.sum(1.0 * (y_pred == y)) / len(y)
    Js_train = np.append(Js_train, J)
    J_test = cost_function(
        theta_k,
        input_layer_size,
        hidden_layer_size,
        num_labels,
        test,
        test_label,
        lmbda,
    )
    test_pred = predict(Theta1, Theta2, test)
    accuracy_test = np.sum(1.0 * (test_pred == test_label)) / len(test_label)
    Js_test = np.append(Js_test, J_test)
    print(
        "iter={:3d}:  Jtrain= {:0.4f} acc= {:0.2f}%  |  Jtest= {:0.4f} acc= {:0.2f}%".format(
            N_iter, J, 100 * accuracy, J_test, 100 * accuracy_test
        )
    )
    N_iter += 1
    if J_test < J_min:
        theta_best = theta_k
        J_min = J_test


def main():
    """Artificial Neural Network for classifying galaxies"""

    global USE_MP, USE_GPU, pool
    parser = argparse.ArgumentParser(
        description="Neural network for galaxy classification"
    )
    parser.add_argument("--mp", action="store_true", help="Enable multiprocessing")
    parser.add_argument("--torch", action="store_true", help="Enable torch")
    args = parser.parse_args()
    USE_MP = args.mp
    USE_GPU = args.torch

    if not USE_GPU and USE_MP:
        pool = mp.Pool()

    np.random.seed(917)
    start = timeit.default_timer()

    # Load the training and test datasets
    train = np.genfromtxt(TRAINING_DIR / "train.csv", delimiter=",")
    test = np.genfromtxt(TRAINING_DIR / "test.csv", delimiter=",")

    # get labels (0=Elliptical, 1=Spiral, 2=Irregular)
    train_label = train[:, 0].reshape(len(train), 1)
    test_label = test[:, 0].reshape(len(test), 1)

    # normalize image data to [0,1]
    train = train[:, 1:] / 255.0
    test = test[:, 1:] / 255.0

    X = train
    y = train_label

    input_layer_size = np.shape(X)[1]
    hidden_layer_size = 8
    num_labels = 3
    lmbda = 1.0  # regularization parameter

    # Initialize random weights:
    Theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1) * 0.4 - 0.2
    Theta2 = np.random.rand(num_labels, hidden_layer_size + 1) * 0.4 - 0.2

    theta0 = np.concatenate((Theta1.flatten(), Theta2.flatten()))
    J = cost_function(
        theta0, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda
    )
    print("initial cost function J =", J)
    train_pred = predict(Theta1, Theta2, train)
    print(
        "initial accuracy on training set =",
        np.sum(1.0 * (train_pred == train_label)) / len(train_label),
    )

    global Js_train, Js_test
    Js_train = np.array([J])
    J_test = cost_function(
        theta0, input_layer_size, hidden_layer_size, num_labels, test, test_label, lmbda
    )
    Js_test = np.array([J_test])

    args = (input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    cbf = partial(
        callbackF,
        input_layer_size,
        hidden_layer_size,
        num_labels,
        X,
        y,
        lmbda,
        test,
        test_label,
    )
    theta = optimize.fmin_cg(
        cost_function, theta0, fprime=gradient, args=args, callback=cbf, maxiter=MAXITER
    )

    Theta1, Theta2 = reshape(
        theta_best, input_layer_size, hidden_layer_size, num_labels
    )

    train_pred = predict(Theta1, Theta2, train)
    test_pred = predict(Theta1, Theta2, test)

    end = timeit.default_timer()
    print("elapsed time = ", end - start)

    print(
        "accuracy on training set =",
        np.sum(1.0 * (train_pred == train_label)) / len(train_label),
    )
    print(
        "accuracy on test set =",
        np.sum(1.0 * (test_pred == test_label)) / len(test_label),
    )

    if USE_MP:
        pool.close()
        pool.join()

    return 0


if __name__ == "__main__":
    main()
