# coding: utf-8
# Shifting from a regression problem to a discrete classification problem.
# Instead of trying to determine the exact color of the pixel, it maps to one colour out of a palette of K colors

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import random
import math
import os


class NeuralNetwork:
    def __init__(self, x, y, num_hidden, epochs, learning_rate, num_nodes_layers, activation_function):
        self.x = x
        self.y = y

        self.n_x = self.x[0].shape[0]  # np.shape(x)[1]  # no. of features   # no. of cols
        self.n_out = self.y[0].shape[0]

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
        self.bias = []



    # parameters: weight and bias
    # weight[l] : (num_layers * num_layers-1 ) * num_layers : (no. of nodes in layer l * no. of nodes in layer (l-1)) * no. of layers
    # bias[l]: () : ( no. of nodes in layer * 1) * no. of layers
    def initialize_parameters_random(self):
        for l in range(1, self.num_layers + 1):
            self.weights.append(
                np.random.rand(self.num_nodes_layers[l], self.num_nodes_layers[l - 1]))


    # Use this when activation function is tanh or sigmoid
    def initialize_parameters_xavier(self):
        for l in range(1, self.num_layers + 1):
            # print("l =", l)
            # print(self.num_nodes_layers[l] , self.num_nodes_layers[l-1])
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
        if self.activation_function == "tanh":
            return np.tanh(x)
        if self.activation_function == "relu":
            a = np.zeros_like(x)
            return np.maximum(a, x)
        if self.activation_function == "leaky_relu":
            a = self.leaky_slope * x
            return np.maximum(a, x)


    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=0, keepdims=True)


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

        for l in range(0, self.num_layers-1):
            z = np.matmul(self.weights[l], A_prev)
            a = self.activation(z)
            A_prev = a
            A.append(a)
            Z.append(z)
        z = np.matmul(self.weights[-1], A_prev)
        a = self.softmax(z)
        A.append(a)
        Z.append(z)
        return (A, Z)


    def loss_function(self, y, out):
        return (0.5 * np.sum((y - out) ** 2))


    def back_propogation(self, A, Z, y):
        delta_z = [None for i in range(self.num_layers)]
        delta_weight = [None for i in range(self.num_layers)]

        delta_z[-1] = (y - A[-1])
        delta_weight[-1] = np.matmul(delta_z[-1], A[-2].T)

        for l in range(self.num_layers - 2, -1, -1):
            delta_z[l] = np.multiply(np.matmul(self.weights[l + 1].T, delta_z[l + 1]), self.gradient_activation(Z[l]))
            delta_weight[l] = np.matmul(delta_z[l], A[l].T)

        return delta_weight


    def update_weight(self, A, delta_weight):
        # weight = weight + learning_rate * error * input
        m = A[-1].shape[1]
        for l in range(self.num_layers):
            self.weights[l] = self.weights[l] + (self.learning_rate * delta_weight[l]) / m


    def predict(self, x_test):
        A, Z = self.forward_propogation(x_test)
        prediction = A[-1]
        return prediction


    def model(self):

        self.initialize_parameters_random()

        for e in range(self.epochs):
            if(e%100 == 0): print("Epoch = ", e)
            loss = []
            n_images = len(self.x)
            for j in range(n_images):
                x = self.x[j]
                y = self.y[j]

                A, Z = self.forward_propogation(x)
                delta_weight = self.back_propogation(A, Z, y)
                self.update_weight(A, delta_weight)

                loss.append(np.sum(-y * np.log(A[-1])))
            print("loss =", loss)




def make_image(output, shape):
    images = np.reshape(output, shape)
    plt.figure()
    plt.imshow(images)


def convert_output_image(img_flattened, img_shape):
    cols = img_flattened.shape[1]
    img = np.full((img_shape[0], img_shape[1], img_shape[2]), 0, dtype="int")
    for i in range(cols):
        row = math.floor(i / img_shape[1])
        col = i % img_shape[1]
        for k in range(img_shape[2]):
            img[row][col][k] = int(img_flattened[k][i])

    return img


def image_to_grayscale(input_image):
    input_image_size = input_image.shape
    num_row = input_image_size[0]
    num_col = input_image_size[1]

    grayscale_image = [
        [0.21 * input_image[row][col][0] + 0.72 * input_image[row][col][1] + 0.07 * input_image[row][col][2] for col in
         range(num_col)] for row in range(num_row)]
    grayscale_image = np.asarray(grayscale_image)
    return grayscale_image


def one_hot_encoding(n, total_classes):

    l =[]
    for i in range(total_classes):
        l.append(0)
    l[n] = 1
    return l

def classifier_ouput(pixel, color_dict):

    diff = np.full((len(color_dict), 1), 1)

    for i in range(len(color_dict)):
        diff[i] = np.mean(np.abs(pixel - color_dict[i]))

    idx = np.argmin(diff)

    return one_hot_encoding(idx, len(color_dict))



def image_to_batch(img, color_dict,  window_size=5, stride=1):
    gray = image_to_grayscale(img)

    padding = int((window_size - 1) / 2)
    padded_img = np.zeros((2 * padding + gray.shape[0], 2 * padding + gray.shape[1]))
    padded_img[padding:-padding, padding:-padding] = gray

    rows = gray.shape[0]
    cols = gray.shape[1]
    input_batch = []
    output_batch = []
    for i in range(0, rows, stride):
        for j in range(0, cols, stride):
            window = padded_img[i:i + window_size, j:j + window_size]
            input_batch.append(window.flatten())
            out = classifier_ouput(img[i,j,:], color_dict)
            output_batch.append(out)
    input_batch = np.asarray(input_batch).T
    output_batch = np.asarray(output_batch).T

    return input_batch, output_batch


def split_data(images, color_dict, window_size=5, stride=1 ):
    train_data = []
    train_output = []
    test_data = []
    test_output = []

    for img in images:
        input_batch, output_batch = image_to_batch(img, color_dict, window_size, stride)

        choice = random.random()
        if choice < 0.8:
            train_data.append(input_batch)
            train_output.append(output_batch)
        else:
            test_data.append(input_batch)
            test_output.append(output_batch)

    return (train_data, train_output), (test_data, test_output)



def get_data(directory_in_str):
    images = []
    directory = os.fsencode(directory_in_str)

    for file in os.listdir(directory):

        path = os.path.join(directory, file)
        filename = os.fsdecode(path)
        img = mpimg.imread(filename)
        images.append(np.asarray(img))

    return images



if __name__ == "__main__":

    THIS_FOLDER = os.path.dirname(__file__)
    my_file = os.path.join(THIS_FOLDER, 'images_dummy')

    images = get_data(my_file)

    color_dict = {0: np.array([255, 0, 0]), 1: np.array([0, 255, 0]), 2: np.array([0, 0, 255]),
                  3: np.array([255, 255, 255]), 4: np.array([0, 0, 0]),
                  5: np.array([255, 255, 0]), 6: np.array([255, 0, 255]),
                  7: np.array([0, 255, 255]), 8: np.array([128, 128, 128]),
                  9: np.array([128, 128, 0]), 10: np.array([128, 0, 128]),
                  11: np.array([0, 128, 128]), 12: np.array([128, 0, 0]),
                  13: np.array([0, 0, 128]), 14: np.array([0, 128, 0]),
                  15: np.array([255, 128, 128]), 16: np.array([128, 255, 128]),
                  17: np.array([128, 128, 255]), 18: np.array([255, 255, 128]),
                  19: np.array([255, 128, 255]), 20: np.array([128, 255, 255]),
                  21: np.array([0, 128, 255]), 22: np.array([128, 255, 0]),
                  23: np.array([255, 0, 128]), 24: np.array([128, 0, 255]),
                  25: np.array([0, 255, 128]), 26: np.array([255, 128, 0]),

                  }

    (x_train, y_train), (test_data, test_output) = split_data(images, color_dict, window_size = 3, stride = 1)


    for i in range(len(x_train)):
        x_train[i] = x_train[i] / 255.0

    nn = NeuralNetwork(x_train, y_train, num_hidden=1, epochs = 10 , learning_rate= 2, num_nodes_layers=[10], activation_function = "sigmoid")
    nn.model()

    for i in range(len(x_train)):
        prediction = nn.predict(x_train[i])
        temp = [color_dict[i] for i in np.argmax(prediction, axis=0)]
        img = convert_output_image(np.asarray(temp).T, images[i].shape)
        plt.figure()
        plt.imshow(img)
        plt.show()


    for i in range(len(x_train)):
        prediction = nn.predict(x_train[i])
        temp = [color_dict[i] for i in np.argmax(prediction, axis=0)]
        img = convert_output_image(np.asarray(temp).T, images[i].shape)
        f = plt.figure()
        plt.imshow(img)


        # Map to K Palette
        f2 = plt.figure()
        temp = [color_dict[i] for i in np.argmax(y_train[i], axis=0)]
        img = convert_output_image(np.asarray(temp).T, images[-1].shape)
        plt.imshow(img)

        # Actual Image
        plt.figure()
        plt.imshow(images[i])
        plt.show()

    for i in range(len(test_output)):
        prediction = nn.predict(test_data[i])
        temp = [color_dict[i] for i in np.argmax(prediction, axis=0)]
        img = convert_output_image(np.asarray(temp).T, images[i].shape)
        f = plt.figure()
        plt.imshow(img)
        f.suptitle("Test - Model Output")

        # Map to K Palette
        f2 = plt.figure()
        temp = [color_dict[i] for i in np.argmax(test_output[i], axis=0)]
        img = convert_output_image(np.asarray(temp).T, images[-1].shape)
        plt.imshow(img)
        f2.suptitle("Test - Actual Output")

        # Actual Image
        plt.figure()
        plt.imshow(images[i])
        plt.show()