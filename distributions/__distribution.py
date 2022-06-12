import numpy as np
import matplotlib.pyplot as plt


class Distribution:

    def __init__(self, dataset):
        """Generic distribution class for calculating and visualizing a probability distribution.

        :param dataset: an 1D array-like numeric dataset.
        """

        self.__data = np.asarray(dataset).flatten()
        self.__mean = self.__std = 0.

    @property
    def data(self):
        return self.__data

    @property
    def mean(self):
        return self.__mean

    @property
    def std(self):
        return self.__std

    @classmethod
    def from_data_file(cls, filename: str, is_sample=True):
        """Return a distribution instance from dataset read from a .txt file.
        The .txt file should have one number (real or integer) per line.

        :param filename: the name or the path of the .txt file containing the dataset.
        :param is_sample: whether the data represents a sample or population. Default is True.
        """

        dataset = []
        with open(filename) as file:
            line = file.readline()
            while line:
                dataset.append(float(line))
                line = file.readline()

        return cls(dataset, is_sample)

    def pdf(self, x: float) -> float: ...

    def plot_histogram(self):
        """Plot the histogram of the dataset."""

        plt.hist(self.data)
        plt.title('Histogram of Data')
        plt.xlabel('data')
        plt.ylabel('count')
        plt.show()

    def plot_pdf(self, n_spaces=50):
        """Plot the normalized histogram and a the probability density function along the same range

        :param n_spaces: number of data points
        """

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

        # return x, y

    def __str__(self):
        """Return the characteristics of the Distribution's instance."""

        return f"Mean: {self.mean} - Standard Deviation: {self.std}"
