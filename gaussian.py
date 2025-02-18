from .__distribution import *
from scipy.integrate import quad


class Gaussian(Distribution):

    def __init__(self, mean: float, std: float):
        """Gaussian distribution class for calculating and visualizing a Gaussian distribution.

        :param mean: the mean of the distribution.
        :param std: the standard deviation of the distribution.
        """

        super().__init__(mean, std)

    @classmethod
    def from_dataset(cls, dataset, is_sample: bool):
        """Return an instance from the input dataset.

        :param dataset: an 1D array-like numeric dataset.
        :param is_sample: whether the data represents a sample or population.
        :return: a new instance of Gaussian distribution class
        """

        if len(dataset) == 0:
            raise ValueError("The input dataset should have at least one element.")

        instance = cls(cls.mean_of(dataset), cls.standard_deviation_of(dataset, is_sample))
        instance._data = dataset

        return instance

    @classmethod
    def from_file(cls, filename: str, is_sample: bool):
        """Return an instance from dataset read from a .txt file.
        The .txt file should have one number (real or integer) per line.

        :param filename: the name or the path of the .txt file containing the dataset.
        :param is_sample: whether the data represents a sample or population.
        :return: a new instance of Gaussian distribution class
        """

        return cls.from_dataset(cls.import_dataset(filename), is_sample)

    @staticmethod
    def mean_of(dataset):
        """Return the mean of the dataset with Gaussian distribution.

        :param dataset: an 1D array-like numeric dataset.
        :return: mean of the dataset.
        """

        return sum(dataset)/len(dataset)

    @staticmethod
    def standard_deviation_of(dataset, is_sample: bool):
        """Return the standard deviation of the dataset with Gaussian distribution.

        :param dataset: an 1D array-like numeric dataset.
        :param is_sample: whether the data represents a sample or population.
        :return: standard deviation of the dataset.
        """

        n = len(dataset) - 1 if is_sample else len(dataset)
        mean_value = Gaussian.mean_of(dataset)
        variance = sum((x - mean_value)**2 for x in dataset) / n
        return math.sqrt(variance)

    def z_score(self, x: float):
        """Return the z-score of the input value with respect to the Gaussian instance.

        A z-score tells how many standard deviations away an value falls from the mean.

        :return: z-score of the input.
        """

        return (x - self.mean) / self.std

    def pdf(self, x: float):
        """Return the result of the value mapped into Probability Density Function
        of the Gaussian instance.

        :param x: a point for calculating the Probability Density Function.
        :return: the output of Probability Density Function.
        """

        return 1 / (self.std * math.sqrt(2*math.pi)) * math.exp(-0.5 * self.z_score(x)**2)

    def probability(self, a: float = -np.inf, b: float = np.inf):
        """Return the probability of a value within a and b (a >= b) in the applied Gaussian distribution.
        If a greater than b, two limits will be swapped.

        :param a: lower limit. Default is negative infinity.
        :param b: upper limit. Default is positive infinity.
        :return: probability of a value within a and b.
        """

        if a == -np.inf and b == np.inf:
            return 1
        elif a == b:
            return 0
        else:
            if a > b:
                a, b = b, a
            return quad(self.pdf, a, b)[0]

    def plot_pdf(self, n_spaces=50):
        """Plot the normalized histogram and the Probability Density Function along the same range.

        :param n_spaces: number of data points
        """

        if len(self.data):
            min_range = min(self.data)
            max_range = max(self.data)

            # calculates the interval between values
            interval = (max_range - min_range) / n_spaces

            x = []
            y = []
            for i in range(n_spaces):
                # Calculate the x to visualize
                tmp = min_range + interval * i
                x.append(tmp)
                y.append(self.pdf(tmp))

            # make the plots
            fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
            fig.subplots_adjust(wspace=0.1)

            # plot the normalized histogram
            axes[0].hist(self.data, density=True)
            axes[0].set_title('Normalized Histogram of Data')
            axes[0].set_ylabel('density')

            # plot the probability density function
            axes[1].plot(x, y)
            axes[1].set_title('Distribution of \n Mean and Standard Deviation')
            
            plt.show()
            # return x, y

        else:
            raise ValueError('Load the dataset first to plot the graphs.')

    def __add__(self, other):
        if type(other) is Gaussian:
            mean = self.mean + other.mean
            std = (self.std**2 + other.std**2)**0.5
            return Gaussian(mean, std)

        else:
            return NotImplemented
