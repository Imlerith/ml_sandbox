from portfolio_trader import *


class LinearModel:
    """ A linear regression model """

    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        # --- momentum terms for coefficient matrix and bias
        self.vW = 0
        self.vb = 0

        # --- losses will be populated on each step of the gradient descent
        self.losses = []

    def predict(self, X):
        assert (len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        assert (len(X.shape) == 2)

        # --- there are N x K values to calculate the loss on
        num_values = np.prod(Y.shape)

        # --- calculate the gradients
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # --- update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # --- update parameters
        self.W += self.vW
        self.b += self.vb

        # --- collect the MSEs
        mse = np.mean((Yhat - Y) ** 2)
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)

