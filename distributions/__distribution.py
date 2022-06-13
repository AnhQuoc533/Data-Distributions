import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import quad


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

    def pdf(self, x: float) -> float: ...

    def plot_histogram(self):
        """Plot the histogram of the dataset."""

        if len(self.data):
            plt.hist(self.data)
            plt.title('Histogram of Data')
            plt.xlabel('data')
            plt.ylabel('count')
            plt.show()

        else:
            raise ValueError('Load the dataset first to plot the graph.')

    def plot_pdf(self, n_spaces=50):
        """Plot the normalized histogram and a the probability density function along the same range

        :param n_spaces: number of data points
        """

        if len(self.data):
            min_range = min(self.data)
            max_range = max(self.data)

            # calculates the interval between x values
            interval = (max_range - min_range) / n_spaces

            x = []
            y = []

            # calculate the x values to visualize
            for i in range(n_spaces):
                tmp = min_range + interval * i
                x.append(tmp)
                y.append(self.pdf(tmp))

            # make the plots
            fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
            fig.subplots_adjust(wspace=0.1)

            # plot the normalized histogram
            axes[0].hist(self.data, density=True)
            axes[0].set_title('Normalized Histogram of Data')
            axes[0].set_ylabel('Density')

            # plot the probability density function
            axes[1].plot(x, y)
            axes[1].set_title('Distribution for \n Mean and Standard Deviation')
            plt.show()

        else:
            raise ValueError('Load the dataset first to plot the graphs.')

        # return x, y

    def __str__(self):
        """Return the characteristics of the Distribution's instance."""

        return f"Mean: {self.mean} - Standard Deviation: {self.std}"
