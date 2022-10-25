import argparse
import numpy as np

from random import uniform
from PIL import Image

class Model:
    @staticmethod
    def sigmoid(x):
        return 1/(np.exp(-x)+1)

    @staticmethod
    def d_sigmoid(x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    def __init__(self, l, s):
        self.__dataset_queue = []

        self.__weights = [
                [uniform(0,1) for x in range(s)] for y in range(l)
            ]
        self.__weights += [uniform(0,1) for x in range(9)]

        self.__biases = [
                [uniform(0,1) for x in range(s)] for y in range(l)
            ]

    def __forward_propagation(self, input_layer):
        current_layer = Model.sigmoid(
                np.dot(input_layer, self.__weights[0]) + self.__biases[0])

        for layer_idx in range(1, len(self.__hidden_layers_weights)):
            current_layer = Model.sigmoid(
                    np.dot(input_layer, self.__weights[layer_idx]
                          ) + self.__biases[layer_idx]
                )

        output_layer = Model.sigmoid(np.dot(current_layer,
                                            self.__output_layer_weights))
        print(output_layer)

    def __backward_propagation(self, output_layer):
        error = (output - expected) * transfer_derivative(output)

    def train(self):
        for image_path in self.__dataset_queue:
            image = Image.open(image_path)
            image_matrix = np.asarray(image)
            # Converting a 3 value matrix (RGB into a single value)
            image_matrix_processed = np.mean(image_matrix / 255,
                                             axis=2).flatten()
            self.__forward_propagation(image_matrix_processed)

    def load_images(self, image_path_list):
        self.__dataset_queue = image_path_list


def main():
    #Inializing the argument parser
    parser = argparse.ArgumentParser()

    # Asking to the user the path of the images
    parser.add_argument('-i', '--image_inputs', nargs='+',
                        help='<Required> Set flag', required=True)
    # Asking to the user the number of layers he wants to use
    parser.add_argument('-l', '--number_hidden_layers', required=True)

    # Asking to the user the number of layers he wants to use
    parser.add_argument('-s', '--size_hidden_layer', required=True)

    # Parsing the arguments
    args = parser.parse_args()

    model = Model(int(args.number_hidden_layers), int(args.size_hidden_layer))
    model.load_images(args.image_inputs)
    model.train()

if __name__ == '__main__':
    main()

