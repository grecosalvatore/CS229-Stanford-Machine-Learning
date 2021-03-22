import numpy as np
import time


class SlowLinearRegression:
    """ Linear Regression class: implementation of a linear regression model with numpy (without using vectorization).

    Attributes:
    """

    def __init__(self):
        self.INIT_PARAMETERS = {"zero", "random"}
        self.OPTIMIZER = {"batch_gradient_descent", "stochastic_gradient_descent"}
        self.theta = None
        return

    def fit(self, X, Y, iterations, learning_rate, init_parameters="zero", optimizer="stochastic_gradient_descent"):
        """
        Fits the linear regression model with the input training examples.

        Args:
            X: input features (n_samples, n_features)
            Y: input labels (n_samples, 1)
            iterations: number of iterations of the training loop
            init_parameters: {"zero","random"} string defining the parameters initiation method
            optimizer: {"batch_gradient_descent","stochastic_gradient_descent"} string with the optimization algorithm
        Returns:
            None
        """

        # First dimension of X,Y (n_samples) must be the same
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Error: first dimension of X {} and Y {} not matches.".format(X.shape[0], Y.shape[0]))

        m = X.shape[0]
        nx = X.shape[1]
        n = nx + 1

        # Add intercept term (1.0 in the first component of each training example)
        X = np.insert(X, 0, 1.0, axis=1)

        # Initialize paramters theta
        self._init_weights(n, init_parameters)

        if optimizer == "batch_gradient_descent":
            self.batch_gradient_descent(X, Y, m, n, iterations, learning_rate)
        if optimizer == "stochastic_gradient_descent":
            self.batch_gradient_descent(X, Y, m, n, iterations, learning_rate)
        return

    def batch_gradient_descent(self, X, Y, m, n, iterations, learning_rate):

        for iteration in range(iterations):
            J = 0
            grads_sum = [0] * n
            # Loop over each training example
            for i in range(m):
                x_i = X[i]
                y_i = Y[i]

                h_i = self._compute_hypothesis(x_i)

                diff_i = h_i - y_i
                loss_i = 1/2*(diff_i)**2

                J += loss_i

                # Loop over each feature
                for j in range(n):
                    grads_sum[j] += diff_i * x_i[j]

            if iteration and iteration % 100 == 0:
                print("Iteration {} - Cost {}".format(iteration, J))

            # Loop over each feature
            for j in range(n):
                self.theta[j] = self.theta[j] - learning_rate * grads_sum[j]

        return

    def stochastic_gradient_descent(self, X, Y, m, n, iterations, learning_rate):

        for iteration in range(iterations):
            J = 0
            # Loop over each training example
            for i in range(m):
                x_i = X[i]
                y_i = Y[i]

                h_i = self._compute_hypothesis(x_i)

                diff_i = h_i - y_i
                loss_i = 1 / 2 * (diff_i) ** 2

                J += loss_i

                # Loop over each feature
                for j in range(n):
                    self.theta[j] = self.theta[j] - learning_rate * (diff_i * x_i[j])

            if iteration and iteration % 100 == 0:
                print("Iteration {} - Cost {:.2f}".format(iteration, J))

        return

    def _compute_hypothesis(self, x_i):
        h = 0
        for j in range(len(self.theta)):
            h += x_i[j] * self.theta[j]
        return h

    def predict(self, x):
        return self._compute_hypothesis(x)

    def evaluate(self):
        return

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


class LinearRegression:
    """ Linear Regression class: implementation of a linear regression model with numpy (without using vectorization).

    Attributes:
    """

    def __init__(self):
        self.INIT_PARAMETERS = {"zero", "random"}
        self.theta = None
        return

    def fit(self, X, Y, iterations, learning_rate, batch_size=512, init_parameters="zero", print_n_iterstions=100):
        """
        Fits the linear regression model with the input training examples.

        Args:
            X: input features (n_samples, n_features)
            Y: input labels (n_samples, 1)
            iterations: number of iterations of the training loop
            init_parameters: {"zero","random"} string defining the parameters initiation method
        Returns:
            None
        """

        # First dimension of X,Y (n_samples) must be the same
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Error: first dimension of X {} and Y {} not matches.".format(X.shape[0], Y.shape[0]))

        m = X.shape[0]
        nx = X.shape[1]
        n = nx + 1

        # Add intercept term (1.0 in the first component of each training example)
        X = np.insert(X, 0, 1.0, axis=1)
        Y = np.reshape(Y, (Y.shape[0], 1))

        # Initialize paramters theta
        self._init_weights(n, init_parameters)

        history, execution_time = self.mini_batch_gradient_descent(X, Y, m, n, batch_size,
                                                                   iterations, learning_rate, print_n_iterstions)

        return history, execution_time

    def mini_batch_gradient_descent(self, X, Y, m, n, batch_size, iterations, learning_rate, print_n_iterstions):
        start_time = time.time()

        history = []
        for iteration in range(iterations):
            J = 0
            n_batch = m % batch_size
            n_batch_remainder = m // batch_size
            # Loop over each training example
            for i_batch in range(n_batch):
                X_batch = X[i_batch:i_batch + batch_size]
                Y_batch = Y[i_batch:i_batch + batch_size]

                H = self._compute_hypothesis(X_batch)

                Diff = H - Y_batch
                J += (1 / (2 * batch_size)) * np.dot(Diff.T, Diff)

                Grad = (1 / batch_size) * np.dot(X_batch.T, Diff)

                self.theta -= learning_rate * Grad

                history.append(J)

            if n_batch_remainder != 0:
                X_batch = X[:-n_batch_remainder]
                Y_batch = Y[:-n_batch_remainder]

                H = self._compute_hypothesis(X_batch)

                Diff = H - Y_batch
                J += (1 / (2 * n_batch_remainder)) * np.dot(Diff.T, Diff)

                Grad = (1 / n_batch_remainder) * np.dot(X_batch.T, Diff)

                self.theta -= learning_rate * Grad

            history.append(J)

            if iteration and iteration % print_n_iterstions == 0:
                print("Iteration {} - Cost {}".format(iteration, J))

        execution_time = time.time() - start_time

        return history, execution_time

    def _compute_hypothesis(self, X):
        if X.shape[1] != self.theta.shape[0]:
            raise ValueError("Error: Input features dimension must be %d." % self.self.theta.shape[0])
        return np.dot(X, self.theta)

    def predict(self, X):
        X = np.insert(X, 0, 1.0, axis=1)
        return self._compute_hypothesis(X)

    def evaluate(self):
        return

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

if __name__ == '__main__':
    m = 100000  # Number of training examples
    nx = 1  # Number of input features

    # Generate X from a random distribution with shape (m, nx)
    X = np.random.rand(m, nx)

    # Generate a noise vector of shape (m, 1)
    epsilon = np.random.normal(0, 0.1, m)

    # Generate Y = (X*2)+2 + noise
    Y = (np.sum(X, axis=1) * 2 + 2) + epsilon

    learning_rate = 0.00001
    iterations = 10000
    init_parameters = "random"  # Random or Zero init
    batch_size = 1024
    LR = LinearRegression()

    history, execution_time = LR.fit(X=X,
                                     Y=Y,
                                     iterations=iterations,
                                     learning_rate=learning_rate,
                                     init_parameters=init_parameters,
                                     batch_size=batch_size)

    print(LR.theta)



