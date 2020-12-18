import numpy as np
from abc import ABC, abstractmethod


class LinearRegression:
    def __init__(self):
        self.n = None  # Number of features
        self.m = None  # Number of training examples
        self.X = None
        self.Y = None
        self.learning_rate = None
        self.theta = None
        self.optimizer = None
        self.history = None
        self.epoches = None
        return

    def model_compile(self, learning_rate=0.01, optimizer=enumerate(["VBGD", "BGD", "SGD", "MBGD"])):
        if optimizer == "VBGD":
            # Vectorized Batch Gradient Descent
            self.optimizer = VectorizedGradientDescentOptimizer(learning_rate)
        elif optimizer == "BGD":
            # Batch Gradient Descent (with explicit for loops)
            self.optimizer = ExplicitForLoopsGradientDescentOptimizer(learning_rate)
        elif optimizer == "SGD":
            # Stochastic Gradient Descent
            self.optimizer = VectorizedGradientDescentOptimizer(learning_rate)
        elif optimizer == "MBGD":
            # Mini-Batch Gradient Descent
            self.optimizer = VectorizedGradientDescentOptimizer(learning_rate)
        else:
            # Default optimization is `vectorized batch gradient descent`
            self.optimizer = VectorizedGradientDescentOptimizer(learning_rate)
        return

    def train(self, X, Y, epoches, init_type=enumerate(["RANDOM", "ZERO"])):
        self.__check_train_parameters(X, Y)  # Check correctness of parameters

        self.m = X.shape[1]  # Number of training examples
        self.n = X.shape[0] + 1  # Number of features (+1 dummy feature x0 = 1)

        self.X = self.__add_dummy_feature(X, self.m, self.n)  # Add dummy feature to each training example
        self.Y = Y

        self.theta = self.__init_weights__(self.n, init_type)
        self.epoches = epoches

        self.theta, self.history = self.optimizer.optimize(self.X, self.Y, self.theta, self.epoches)

        return self.theta

    def predict(self, X):
        X = np.array(X)
        new_X = self.__add_dummy_feature(X, X.shape[1], X.shape[0]+1)
        hypothesis = np.dot(self.theta.T, new_X)
        return hypothesis

    @staticmethod
    def __add_dummy_feature(X, m, n):
        new_X = np.insert(X, 0, 1., axis=0)
        return new_X

    @staticmethod
    def __init_weights__(n_features, init_type=enumerate(["RANDOM", "ZERO"])):
        if init_type == "RANDOM":
            # Initialize weights `theta` with random values
            theta = np.random.rand(n_features, 1)
        elif init_type == "ZERO":
            # Initialize weights `theta` with zero values
            theta = np.zeros((n_features, 1), dtype=float)
        else:
            # Default initiliziation is `zero`
            theta = np.zeros((n_features, 1), dtype=float)
        return theta

    @staticmethod
    def __check_train_parameters(X, Y):
        if not isinstance(X, np.ndarray):
            raise ValueError("ValueError: X must be of type np.ndarray.")
        if not isinstance(Y, np.ndarray):
            raise ValueError("ValueError: Y must be of type np.ndarray.")
        if X.ndim != 2:
            raise ValueError(
                "ValueError: X must have two dimensions (m, n) (number of training examples, number of features).")
        if Y.ndim != 2:
            raise ValueError(
                "ValueError: X must have two dimensions (m, n) (number of training examples, number of features).")
        if X.shape[1] != Y.shape[1]:
            raise ValueError("ValueError: second axes (m = number training examples) of X and Y must match.")
        if X.shape[0] < 1:
            raise ValueError("ValueError: first axes (n = number of features) of X must be > 1.")
        if Y.shape[0] != 1:
            raise ValueError("ValueError: first axes of Y must be = 1.")
        return


class LinearRegressionOptimizer(ABC):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.history = []
        return

    @abstractmethod
    def optimize(self, X, Y, theta, epochs):
        pass

    @abstractmethod
    def compute_hypothesis(self):
        pass

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def compute_cost_function(self):
        pass

    @abstractmethod
    def compute_gradient(self):
        pass

    @abstractmethod
    def update_parameter(self):
        pass


class VectorizedGradientDescentOptimizer(LinearRegressionOptimizer):
    def __init__(self, learning_rate):
        LinearRegressionOptimizer.__init__(self, learning_rate)
        return

    def optimize(self, X, Y, theta, epochs):
        for e in range(epochs):

            hypothesis = self.compute_hypothesis(X, theta)
            print("H(theta): ", hypothesis.shape)
            losses, H_minus_Y = self.compute_loss(hypothesis, Y)
            print("H_minus_Y: ", H_minus_Y.shape)
            print("losses: ", losses.shape)
            cost = self.compute_cost_function(losses)
            print("cost: ", cost.shape)
            gradients = self.compute_gradient(X, H_minus_Y)
            print("gradients: ", gradients.shape)
            theta = self.update_parameter(theta, gradients, self.learning_rate)
            print("theta: ", theta.shape)
            self.history.append(cost)

        return theta, self.history

    @staticmethod
    def compute_hypothesis(X, theta):
        hypothesis = np.dot(theta.T, X)
        return hypothesis

    @staticmethod
    def compute_loss(hypothesis, Y):
        H_minus_Y = hypothesis - Y
        return np.square(H_minus_Y), H_minus_Y

    @staticmethod
    def compute_cost_function(losses):
        J = np.sum(losses) / 2
        return J

    @staticmethod
    def compute_gradient(X, H_minus_Y):
        return np.dot(X, H_minus_Y.T)

    @staticmethod
    def update_parameter(theta, gradients, learning_rate):
        return theta - learning_rate*gradients


class ExplicitForLoopsGradientDescentOptimizer(LinearRegressionOptimizer):
    def __init__(self, learning_rate):
        LinearRegressionOptimizer.__init__(self, learning_rate)
        return

    def optimize(self, X, Y, theta, epochs):
        m = X.shape[1]
        n = X.shape[0]
        for e in range(epochs):
            cost = 0
            gradients = [0]*n
            for i in range(m):
                x_i = X[:,i]
                y_i = Y[:,i]
                h_i = self.compute_hypothesis(x_i, theta, n)
                loss_i = self.compute_loss(h_i, y_i)

                cost += loss_i

                for j in range(n):
                    gradient_j = self.compute_gradient(h_i, y_i, x_i[j])
                    gradients[j] += gradient_j

            for j in range(n):
                theta[j] = self.update_parameter(theta[j], gradients[j], self.learning_rate)
            cost = cost / 2
            self.history.append(cost)


        return theta, self.history

    @staticmethod
    def compute_hypothesis(x_i, theta, n):
        h_i = 0
        for j in range(n):
            h_i += x_i[j] * theta[j]
        return h_i

    @staticmethod
    def compute_loss(h_i, y_i):
        loss_i = (h_i - y_i)*(h_i - y_i)
        return loss_i

    @staticmethod
    def compute_cost_function(losses):
        J = np.sum(losses) / 2
        return J

    @staticmethod
    def compute_gradient(h_i, y_i, x_j):
        return (h_i-y_i)*x_j

    @staticmethod
    def update_parameter(theta_j, gradient_j, learning_rate):
        return theta_j - learning_rate*gradient_j


def run_linear_regression():
    m = 300
    n = 10

    X = np.random.rand(n, m)
    X = np.array([[2,2],[5,6],[0,10]])
    X = X.T
    print(X.shape)

    Y = np.random.rand(1, m)
    Y = np.array([[5],[12],[11]])
    #Y = np.sum(X, axis=0)
    Y = Y.T
    print(Y.shape)

    lr = LinearRegression()
    lr.model_compile(learning_rate=0.01, optimizer="VGD")
    theta = lr.train(X, Y, 10000, init_type="ZERO")
    print(theta)
    #print(lr.predict([3,2,4,5,0]))
    print("")
    return

if __name__ == '__main__':
    run_linear_regression()


