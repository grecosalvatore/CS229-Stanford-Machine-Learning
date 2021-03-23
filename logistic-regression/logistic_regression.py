import pandas as pd
import numpy as np

import os
import time

class LogisticRegression:

    def __init__(self):
        self.INIT_PARAMETERS = {"zero", "random"}
        self.theta = None
        return

    def fit(self, X, Y, iterations, learning_rate, batch_size=512, init_parameters="zero", step_per_iterations=10):

        # First dimension of X,Y (n_samples) must be the same
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Error: first dimension of X {} and Y {} not matches.".format(X.shape[0], Y.shape[0]))

        m = X.shape[0]
        nx = X.shape[1]
        n = nx + 1

        # Add 1 in first component of all Xs (interpect term)
        X = np.insert(X, 0, 1.0, axis=1)
        Y = np.reshape(Y, (Y.shape[0], 1))

        # Initialize paramters theta
        self._init_weights(n, init_parameters)

        history, execution_time = self.mini_batch_gradient_descent(X, Y, m, n, batch_size,
                                                                   iterations, learning_rate, step_per_iterations)

        print("Training takes {:.2f} seconds.".format(execution_time))
        return history

    def mini_batch_gradient_descent(self, X, Y, m, n, batch_size, iterations, learning_rate, step_per_iterations):
        start_time = time.time()

        history = []

        # Training loop
        for iteration in range(iterations):

            J = 0
            n_batch = m // batch_size
            n_batch_remainder = m % batch_size

            for i_batch in range(n_batch):
                X_batch = X[i_batch:i_batch + batch_size]
                Y_batch = Y[i_batch:i_batch + batch_size]

                # Forward Propagation
                J_batch, Diff = self.forward_propagation(X_batch, Y_batch, batch_size)

                J += J_batch

                # Backward Propagation
                self.backward_propagation(X_batch, Diff, batch_size, learning_rate)

                history.append(J)

            if n_batch_remainder != 0:
                X_batch = X[:-n_batch_remainder]
                Y_batch = Y[:-n_batch_remainder]

                # Forward Propagation
                J_batch, Diff = self.forward_propagation(X_batch, Y_batch, n_batch_remainder)

                J += J_batch

                # Backward Propagation
                self.backward_propagation(X_batch, Diff, n_batch_remainder, learning_rate)

                history.append(J)

            if iteration and iteration % step_per_iterations == 0:
                print("Iteration {} - Cost {}".format(iteration, J))

        execution_time = time.time() - start_time

        return history, execution_time

    def forward_propagation(self, X, Y, m):
        # Compute the hypothesis for all Xs in the batch
        H = self._compute_hypothesis(X)

        # Compute the difference between estimated y and actual label
        Diff = Y - H

        # Sum the squared differences
        J = (1 / (2 * m)) * np.dot(Diff.T, Diff)
        return J, Diff

    def backward_propagation(self, X, Diff, m, learning_rate):
        Grad = (1 / m) * np.dot(X.T, Diff)

        # Update paramters theta
        self.theta += learning_rate * Grad
        return

    def predict(self, X):
        # Add 1 in first component of all Xs (interpect term)
        X = self._add_intercept(X)
        return self._compute_hypothesis(X)

    def _compute_hypothesis(self, X):
        Z = np.dot(X, self.theta)
        return self.sigmoid(Z)

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def _add_intercept(X):
        # Add intercept term (1.0 in the first component of each input)
        return np.insert(X, 0, 1.0, axis=1)

    def _init_weights(self, n, init_parameters="zero"):
        """ initiates the paramters as zero or random. """
        if init_parameters not in self.INIT_PARAMETERS:
            raise ValueError("Error: init_parameters must be one of %s." % self.INIT_PARAMETERS)

        if init_parameters == "zero":
            # Initialize paramters with zero values
            self.theta = np.zeros((n, 1), dtype=float)

        if init_parameters == "random":
            # Initialize paramters with random values
            self.theta = np.random.rand(n, 1)
        return

def load_dataset():
    DATASET_FOLDER = os.path.join("..", "datasets", "logistic-regression-data")

    df_x = pd.read_csv(os.path.join(DATASET_FOLDER, "train_inputs.txt"), sep="\ +", names=["x1", "x2"], header=None,
                       engine='python')
    df_y = pd.read_csv(os.path.join(DATASET_FOLDER, "train_labels.txt"), sep='\ +', names=["y"], header=None,
                       engine='python')

    df_y["y"] = df_y.apply(lambda row: max(row["y"], 0), axis=1)

    X = df_x[["x1", "x2"]].values
    Y = df_y["y"].values

    return X, Y

def run():
    learning_rate = 0.01
    iterations = 100
    init_parameters = "random"  # Random or Zero init
    batch_size = 8

    X, Y = load_dataset()

    LR_model = LogisticRegression()

    history = LR_model.fit(X=X,
                           Y=Y,
                           iterations=iterations,
                           learning_rate=learning_rate,
                           init_parameters=init_parameters,
                           batch_size=batch_size,
                           step_per_iterations=1)
    return

if __name__ == '__main__':
    run()