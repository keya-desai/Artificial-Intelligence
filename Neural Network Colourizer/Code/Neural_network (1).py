# -*- coding: utf-8 -*-
# Colourizer - Approach 1 #
# Direct Mapping from gray scale values to (r,g,b) values of every pixel. #
# The input can be images of any size but they need to be of the same length #


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import random
import os


class NeuralNetwork:
    def __init__(self, x, y, num_hidden, epochs, learning_rate, num_nodes_layers, activation_function, batch_size):
        self.x = x
        self.y = y

        self.num_data = np.shape(x)[1]  # no. of data points    # no. of rows
        self.n_x = np.shape(x)[0]  # no. of features   # no. of cols
        self.n_out = np.shape(y)[0]

        self.batch_size = batch_size
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_hidden = num_hidden
        self.num_layers = num_hidden + 1  # +1 for output layer

        self.num_nodes_layers = num_nodes_layers
        # inserting input and output nodes to the list
        self.num_nodes_layers.insert(0, self.n_x)
        self.num_nodes_layers.append(self.n_out)

        self.leaky_slope = 0.01
        self.weights = []

    # parameters: weight and bias
    # weight[l] : (num_layers * num_layers-1 ) * num_layers : (no. of nodes in layer l * no. of nodes in layer (l-1)) * no. of layers
    def initialize_parameters_random(self):

        for l in range(1, self.num_layers + 1):
            self.weights.append(
                np.random.rand(self.num_nodes_layers[l], self.num_nodes_layers[l - 1]))

    # Use this when activation function is tanh or sigmoid
    def initialize_parameters_xavier(self):

        for l in range(1, self.num_layers + 1):
            self.weights.append(np.random.randn(self.num_nodes_layers[l], self.num_nodes_layers[l - 1]) * np.sqrt(
                1 / self.num_nodes_layers[l - 1]))

    # Use this when activation function is ReLU or Leaky ReLu
    def initialize_parameters_he(self):
        for l in range(1, self.num_layers + 1):
            self.weights.append(np.random.randn(self.num_nodes_layers[l], self.num_nodes_layers[l - 1]) * np.sqrt(
                2 / self.num_nodes_layers[l - 1]))

    # Activation Functions
    def activation(self, x):
        if self.activation_function == "linear":
            return x
        if self.activation_function == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation_function == "tanh":
            return np.tanh(x)
        elif self.activation_function == "relu":
            a = np.zeros_like(x)
            return np.maximum(a, x)
        elif self.activation_function == "leaky_relu":
            a = self.leaky_slope * x
            return np.maximum(a, x)

    def gradient_activation(self, X):
        if self.activation_function == "linear":
            return np.ones_like(X)
        elif self.activation_function == "sigmoid":
            return self.activation(X) * (1 - self.activation(X))
        elif self.activation_function == "tanh":
            return (1 - np.square(X))
        elif self.activation_function == "relu":
            grad = np.zeros_like(X)
            grad[X > 0] = 1.0
            return grad
        elif self.activation_function == "leaky_relu":
            grad = np.ones_like(X)
            grad[X <= 0] = self.leaky_slope
            return grad

    def forward_propogation(self, x):
        # dim of A vector: (no. of hidden nodes * num_data) *(no. of layers)
        A = []
        Z = []
        A.append(x)
        A_prev = x

        for l in range(0, self.num_layers):
            z = np.matmul(self.weights[l], A_prev)
            a = self.activation(z)
            A_prev = a
            A.append(a)
            Z.append(z)
        return (A, Z)


    def back_propogation(self, A, Z, y):

        delta_z = [None for i in range(self.num_layers)]
        delta_weight = [None for i in range(self.num_layers)]

        delta_z[-1] = (y - A[-1])
        delta_weight[-1] = np.matmul(delta_z[-1], A[-2].T)

        for l in range(self.num_layers - 2, -1, -1):
            delta_z[l] = np.multiply(np.matmul(self.weights[l + 1].T, delta_z[l + 1]), self.gradient_activation(Z[l]) )
            delta_weight[l] = np.matmul( delta_z[l], A[l].T )

        return delta_weight


    def update_weight(self, A, delta_weight):
        # weight = weight + learning_rate * error * input
        m = A[-1].shape[1]
        for l in range(self.num_layers):
            self.weights[l] = self.weights[l] + (self.learning_rate * delta_weight[l])/m

    def predict(self, x_test):
        A,Z = self.forward_propogation(x_test)
        prediction = A[-1]

        return prediction


    def loss_function(self, y, out):
        return (0.5 * np.mean((y - out) ** 2))


    def model(self):

        mini_batch = int((self.num_data) / (self.batch_size))
        self.initialize_parameters_random()

        for e in range(self.epochs):

            print("Epoch =", e)
            end = 0
            for n in range(mini_batch + 1):

                if (n != mini_batch):
                    start = n * self.batch_size
                    end = (n + 1) * self.batch_size
                    x_ = self.x[:, start:end]
                    y_ = self.y[:, start:end]

                else:
                    if ((self.num_data % self.batch_size) != 0):
                        x_ = self.x[:, end:]
                        y_ = self.y[:, end:]
                    else:
                        break

                A,Z = self.forward_propogation(x_)
                delta_weight = self.back_propogation(A, Z, y_)
                self.update_weight(A, delta_weight)

                loss = self.loss_function(A[-1]*255, y_*255)

            print("loss = ", loss)


def make_image(output, shape):
    images = np.reshape(output, shape)
    plt.figure()
    plt.imshow(images)


def image_to_grayscale(input_image):
    input_image_size = input_image.shape
    num_row = input_image_size[0]
    num_col = input_image_size[1]

    grayscale_image = [
        [0.21 * input_image[row][col][0] + 0.72 * input_image[row][col][1] + 0.07 * input_image[row][col][2] for col in
         range(num_col)] for row in range(num_row)]
    grayscale_image = np.asarray(grayscale_image)
    return grayscale_image


def get_data(directory_in_str):
    train_data = []
    train_output = []
    validation_data = []
    validation_output = []
    test_data = []
    test_output = []

    directory = os.fsencode(directory_in_str)

    for file in os.listdir(directory):

        path = os.path.join(directory, file)
        filename = os.fsdecode(path)

        img = mpimg.imread(filename)
        gray = image_to_grayscale(img)
        img = np.asarray(img)
        image_shape = img.shape
        img = img.flatten()


        gray = gray.flatten()
        choice = random.random()
        if choice < 1:
            train_data.append(gray)
            train_output.append(img)

        elif choice < 0.9:
            validation_data.append(gray)
            validation_output.append(img)

        else:
            test_data.append(gray)
            test_output.append(img)

    train_data = np.asarray(train_data).T
    train_output = np.asarray(train_output).T
    validation_data = np.asarray(validation_data).T
    validation_output = np.asarray(validation_output).T
    test_data = np.asarray(test_data).T
    test_output = np.asarray(test_output).T

    return (train_data, train_output), (validation_data, validation_output), (test_data, test_output), image_shape


if __name__ == "__main__":

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, 'images_dummy')
    (x_train, y_train), (validation_data, validation_output), (test_data, test_output), image_shape = get_data(my_file)

    x_train = x_train/255.
    y_train = y_train/255.

    nn = NeuralNetwork(x_train, y_train, num_hidden= 1, epochs= 1, learning_rate=0.5, num_nodes_layers=[10],
                       activation_function="sigmoid", batch_size = 1)
    nn.model()

    for i in range(x_train.shape[1]):
        if(i>3):
            break
        prediction = nn.predict(x_train[:, i])
        make_image((prediction * 255).astype(int), image_shape)

        # Actual Image
        make_image(y_train[:, i], image_shape)
        plt.show()