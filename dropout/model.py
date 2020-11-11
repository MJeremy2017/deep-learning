import numpy as np


class deepNN:
    def __init__(self, layers):
        self.layers = layers
        self.params = {}
       
    
    def weights_init(self):
        n = len(self.layers)
        for i in range(1, n):
            self.params['W' + str(i)] = np.random.randn(self.layers[i], self.layers[i-1])*0.01
            self.params['b' + str(i)] = np.zeros((self.layers[i], 1))
    
    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)
    
    @staticmethod
    def compute_cost(A, Y):
        """
        For binary classification, both A and Y would have shape (1, m), where m is the batch size
        """
        assert A.shape == Y.shape
        m = A.shape[1]
        s = np.dot(Y, np.log(A.T)) + np.dot(1-Y, np.log((1 - A).T))
        loss = -s/m
        return np.squeeze(loss)
    
    @staticmethod
    def sigmoid_grad(A, Z):
        grad = np.multiply(A, 1-A)
        return grad

    @staticmethod
    def relu_grad(A, Z):
        grad = np.zeros(Z.shape)
        grad[Z>0] = 1
        return grad
    
    
    def forward(self, X):
        # intermediate layer use relu as activation
        # last layer use sigmoid
        n_layers = int(len(self.params)/2)
        A = X
        cache = {}
        for i in range(1, n_layers):
            W, b = self.params['W'+str(i)], self.params['b'+str(i)]
            Z = np.dot(W, A) + b
            A = self.relu(Z)
            cache['Z'+str(i)] = Z
            cache['A'+str(i)] = A

        # last layer
        W, b = self.params['W'+str(i+1)], self.params['b'+str(i+1)]
        Z = np.dot(W, A) + b
        A = self.sigmoid(Z)
        cache['Z'+str(i+1)] = Z
        cache['A'+str(i+1)] = A

        return cache, A
    
    def backward(self, cache, X, Y):
        """
        cache: result [A, Z]
        Y: shape (1, m)
        """
        grad = {}
        n_layers = int(len(self.params)/2)
        m = Y.shape[1]
        cache['A0'] = X

        for l in range(n_layers, 0, -1):
            A, A_prev, Z = cache['A' + str(l)], cache['A' + str(l-1)], cache['Z' + str(l)]
            W = self.params['W'+str(l)]
            if l == n_layers:
                dA = -np.divide(Y, A) + np.divide(1 - Y, 1 - A)

            if l == n_layers:
                dZ = np.multiply(dA, self.sigmoid_grad(A, Z))
            else:
                dZ = np.multiply(dA, self.relu_grad(A, Z))
            dW = np.dot(dZ, A_prev.T)/m
            db = np.sum(dZ, axis=1, keepdims=True)/m
            dA = np.dot(W.T, dZ)

            grad['dW'+str(l)] = dW
            grad['db'+str(l)] = db

        return grad
    
    def optimize(self, grads, lr):
        n_layers = int(len(self.params)/2)
        for i in range(1, n_layers+1):
            dW, db = grads['dW'+str(i)], grads['db'+str(i)]
            self.params['W'+str(i)] -= lr*dW
            self.params['b'+str(i)] -= lr*db
    
    @staticmethod
    def generate_batch(X, batch_size):
        n = X.shape[0]
        batches = [range(i, i+batch_size) for i in range(0, n, batch_size)]
        return batches
    
    def train(self, X_train, y_train, batch_size=200, n_iter=100, lr=0.1):
        # prepare batch training
        batches = self.generate_batch(X_train, batch_size)
        # init weights
        self.weights_init()
        for i in range(n_iter):
            for batch in batches:
                X = X_train[batch, :].T
                Y = y_train[batch].reshape(1, -1)
                cache, A = self.forward(X)
                grads = self.backward(cache, X, Y)
                self.optimize(grads, lr)

            if i%10 == 0:
                loss = self.compute_cost(A, Y)
                print(f'iteration {i}: loss {loss}')


def accuracy(Y, Y_pred):
    """
    Y: vector of true value
    Y_pred: vector of predicted value
    """
    
    assert Y.shape[0] == 1
    assert Y.shape == Y_pred.shape
    Y_pred = np.round(Y_pred)
    acc = float(np.dot(Y, Y_pred.T) + np.dot(1 - Y, 1 - Y_pred.T))/Y.size
    return acc