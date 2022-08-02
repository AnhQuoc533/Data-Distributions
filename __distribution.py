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

    @property
    def mean(self):
        return self.__mean

    @property
    def std(self):
        return self.__std

    def plot_histogram(self):
        """Plot the histogram of the dataset."""

        if len(self.data):
            plt.hist(self.data)
            plt.title('Histogram of Data')
            plt.xlabel('data')
            plt.ylabel('count')
            plt.show()

        else:
            raise ValueError('Load the dataset to the instance first to plot the graph.')

    def __str__(self):
        """Return the characteristics of the Distribution's instance."""

        return f"Mean: {self.mean} - Standard Deviation: {self.std}"
