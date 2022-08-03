import matplotlib.pyplot as plt
import math
import numpy as np


class Distribution:

    def __init__(self, mean: float, std: float):
        """Generic distribution class for calculating and visualizing a probability distribution.

        :param mean: the mean of the distribution.
        :param std: the standard deviation of the distribution.
        """

        self._data = []
        self.__mean = mean
        self.__std = std

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, dataset): ...

    @property
    def mean(self):
        return self.__mean

    @property
    def std(self):
        return self.__std

    @staticmethod
    def import_dataset(filename: str):
        """Return a list of numbers read from a .txt file.
        The .txt file should have one number (real or integer) per line.

        :param filename: the name or the path of the .txt file containing the dataset.
        :return: a list of numbers
        """

        dataset = []
        with open(filename) as file:
            for line in file:
                dataset.append(eval(line))

        return dataset

    def plot_histogram(self):
        """Plot the histogram of the dataset."""

        if len(self._data):
            plt.hist(self._data)
            plt.title('Histogram')
            plt.xlabel('data')
            plt.ylabel('count')
            plt.show()

        else:
            raise ValueError('Load the dataset to the instance first to plot the graph.')

    def __str__(self):
        """Return the characteristics of the Distribution's instance."""

        return f"Mean: {self.mean} - Standard Deviation: {self.std}"
