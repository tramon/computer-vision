import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import random


class NumberNeuroIdentifier:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weights init
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

        # parameters for Adam
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.beta1, self.beta2 = 0.9, 0.999
        self.epsilon = 1e-8
        self.t = 0

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, Y_pred, Y_true):
        m = Y_true.shape[0]
        log_likelihood = -np.log(Y_pred[range(m), Y_true])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, X, Y):
        m = X.shape[0]
        self.t += 1

        dZ2 = self.A2
        dZ2[range(m), Y] -= 1
        dZ2 /= m

        dW2 = self.A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Adam updates
        for param, dparam, m, v in [
            (self.W1, dW1, self.mW1, self.vW1),
            (self.b1, db1, self.mb1, self.vb1),
            (self.W2, dW2, self.mW2, self.vW2),
            (self.b2, db2, self.mb2, self.vb2)
        ]:
            m[:] = self.beta1 * m + (1 - self.beta1) * dparam
            v[:] = self.beta2 * v + (1 - self.beta2) * (dparam ** 2)
            m_corr = m / (1 - self.beta1 ** self.t)
            v_corr = v / (1 - self.beta2 ** self.t)
            param -= self.learning_rate * m_corr / (np.sqrt(v_corr) + self.epsilon)

    def train(self, X, Y, epochs=100):
        loss_history = []
        for i in range(epochs):
            Y_pred = self.forward(X)
            loss = self.compute_loss(Y_pred, Y)
            loss_history.append(loss)
            self.backward(X, Y)
            if i % 10 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

        plt.figure(figsize=(8, 4))
        plt.plot(loss_history, label="Loss", color='cyan')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.grid(True)
        plt.legend()
        plt.show()

    def predict(self, X):
        Y_pred = self.forward(X)
        return np.argmax(Y_pred, axis=1)

    def visualize_input(self, X, labels):
        num_samples = len(X)
        cols = num_samples
        plt.figure(figsize=(cols * 2, 2))
        for i in range(num_samples):
            plt.subplot(1, cols, i + 1)
            plt.imshow(X[i].reshape(8, 8), cmap='gray')
            plt.title(str(labels[i]))
            plt.axis('off')
        plt.suptitle("Number to learn")
        plt.tight_layout()
        plt.show()

    def predict_random_sample(self, X, Y):
        index = random.randint(0, len(X) - 1)
        sample = X[index].reshape(1, -1)
        label = Y[index]
        prediction = self.predict(sample)[0]

        plt.imshow(sample.reshape(8, 8), cmap='gray')
        plt.title(f"Predicted: {prediction}, Actual: {label}")
        plt.axis('off')
        plt.show()

        print(f"Model prediction: {prediction} (Actual: {label})")


if __name__ == '__main__':
    digits = load_digits()

    mask = np.ones_like(digits.target, dtype=bool)
    X = digits.data[mask] / 16.0
    Y = digits.target[mask]

    hidden_size = 8
    learning_rate = 0.1
    epochs = 70

    model = NumberNeuroIdentifier(input_size=64, hidden_size=hidden_size, output_size=10,
                                  learning_rate=learning_rate)
    model.visualize_input(X[:10], Y[:10])
    model.train(X, Y, epochs=epochs)

    predictions = model.predict(X)
    print("Prediction:", predictions[:10])
    print("Actual:     ", Y[:10])

    model.predict_random_sample(X, Y)
