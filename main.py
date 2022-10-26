import os
import sys
import argparse
import numpy as np
import math
import typing

from random import uniform
from PIL import Image

class Model:
    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def d_sigmoid(z):
        return Model.sigmoid(z)*(1-Model.sigmoid(z))

    @staticmethod
    def cost(outputs_activations, y):
        return (outputs_activations-y)

    @staticmethod
    def softmax(x):
        exp_values = np.exp(np.mean(x) - np.max(x, axis=1, keepdims=True))
        prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return prob

    @staticmethod
    def vectorize(n):
        e = np.zeros((10, 1))
        e[n] = 1.0
        return e

    def __init__(self, l, s):
        self.__dataset_queue = []

        sizes = [784, 784, 784, 10]
        self.__nb_layers = len(sizes)
        self.__biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.__weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def __forward_propagation(self, layer):
        for w, b in zip(self.__weights, self.__biases):
            layer = Model.sigmoid(np.dot(w, layer) + b)
        return layer

    def __backward_propagation(self, x, y):
        nab_b = [np.zeros(b.shape) for b in self.__biases]
        nab_w = [np.zeros(w.shape) for w in self.__weights]

        current_layer_activations = x
        layers_activations_list   = [x]
        layers_values_list        = []

        # Forward propagation
        for b, w in zip(self.__biases, self.__weights):
            layer_values = np.dot(w, current_layer_activations) + b
            layers_values_list.append(layer_values)

            current_layer_activations = Model.sigmoid(layer_values)
            layers_activations_list.append(current_layer_activations)

        delta = Model.cost(layers_activations_list[-1], y) * \
            Model.d_sigmoid(layers_values_list[-1])
        nab_b[-1] = delta
        nab_w[-1] = np.dot(delta, layers_activations_list[-2].T)

        # Backward propagation
        for layer_idx in range(2, self.__nb_layers):
            layer_values = layers_values_list[-layer_idx]
            derivative = Model.sigmoid(layer_values)
            delta = np.dot(
                self.__weights[-layer_idx+1].T,
                delta
            ) * derivative

            nab_b[-layer_idx] = delta
            nab_w[-layer_idx] = np.dot(
                delta,
                layers_activations_list[-layer_idx-1].T
            )

        return (nab_b, nab_w)

    def __update(self, x, y, learning_rate):
        nabla_b, nabla_w = self.__backward_propagation(x, y)

        self.weights = [w - learning_rate * nab_w
                        for w, nab_w in zip(self.__weights, nabla_w)]
        self.biases = [b - learning_rate * nab_b
                       for b, nab_b in zip(self.__biases, nabla_b)]

    def train(self, learning_rate: float, epoch: int):
        for image_path, y in self.__dataset_queue:
            sys.stdout.write("\rProcessing file: {}".format(image_path))
            sys.stdout.flush()
            with Image.open(image_path) as image:
                image_matrix = np.asarray(image)
                # Converting a 3 value matrix (RGB into a single value)
                image_matrix_processed = np.mean(image_matrix / 255,
                                                 axis=2).flatten()
                for i in range(epoch):
                    self.__update(image_matrix_processed, y, learning_rate)

    def test(self, image_path: str):
        with Image.open(image_path) as image:
            image_matrix = np.asarray(image)
            # Converting a 3 value matrix (RGB into a single value)
            image_matrix_processed = np.mean(image_matrix / 255,
                                             axis=2).flatten()
            pred = self.__forward_propagation(image_matrix_processed)
            print()
            print(np.argmax(pred))

    def load_images(self, folder_path: str, label: int):
        y = Model.vectorize(label)
        self.__dataset_queue.append((folder_path, y))

def main():
    #Inializing the argument parser
    parser = argparse.ArgumentParser()

    # Asking to the number of files to load
    parser.add_argument('-n', '--nb_files', required=True, type=int)

    # Asking to the user the number of layers he wants to use
    parser.add_argument('-l', '--number_hidden_layers', required=True,
                        type=int)

    # Asking to the user the number of layers he wants to use
    parser.add_argument('-s', '--size_hidden_layer', required=True,
                        type=int)

    # Asking to the user the learning rate
    parser.add_argument('-r', '--learning_rate', nargs='?', const=1,
                        type=float, default=0.01)

    # Asking to the user the number of epoch
    parser.add_argument('-e', '--epoch', nargs='?', const=1,
                        type=int, default=1000)

    # Parsing the arguments
    args = parser.parse_args()

    dataset = [
        ('train_dataset/2', 2),
        ('train_dataset/4', 4),
        ('train_dataset/5', 5),
        ('train_dataset/6', 6),
        ('train_dataset/7', 7),
    ]
    model = Model(int(args.number_hidden_layers), int(args.size_hidden_layer))
    for path, label in dataset:
        for filename in os.listdir(path)[:args.nb_files]:
            full_path = os.path.join(path, filename)
            model.load_images(full_path, label)

    model.train(learning_rate=args.learning_rate, epoch=args.epoch)
    model.test('./train_dataset/2/0001.png')

if __name__ == '__main__':
    main()

