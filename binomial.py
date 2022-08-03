from .__distribution import *


class Binomial(Distribution):

    def __init__(self, size: int, prob: float):
        """Binomial distribution class for calculating and visualizing a binomial distribution.

        :param size: the number of trials.
        :param prob: the success probability for each trial.
        """

        if size < 1 or type(size) is not int:
            raise ValueError("Positive integers expected for the size parameter.")
        elif not (0 <= prob <= 1):
            raise ValueError("The probability must be between 0 and 1.")

        self.__n = size
        self.__p = prob
        self.__q = 1 - prob
        super().__init__(mean=self.n*self.p, std=(self.n*self.p*self.q)**0.5)

    @property
    def p(self):
        return self.__p

    @property
    def q(self):
        return self.__q

    @property
    def n(self):
        return self.__n

    @classmethod
    def from_binary_data(cls, dataset):
        """Return a binomial instance built from the input binary dataset (contains only 1 and 0)
        or (True and False).

        :param dataset: an 1D array-like binary dataset.
        :return: a new instance of Binomial class
        """

        if len(dataset) == 0:
            raise ValueError("The input dataset should have at least one element.")
        elif set(dataset) == {True, False}:
            dataset = [int(x) for x in dataset]

        if set(dataset) == {0, 1}:
            instance = Binomial(len(dataset), sum(dataset) / len(dataset))
            instance._data = dataset
            return instance
        else:
            raise ValueError("A binary dataset (contains only 1 and 0) expected.")

    @classmethod
    def from_file(cls, filename: str):
        """Return a binomial instance built from the binary dataset (contains only 1 and 0) read from .txt file.
        The .txt file should have one number per line.

        :param filename: the name or the path of the .txt file containing the dataset.
        :return: a new instance of Binomial class
        """

        return cls.from_binary_data(cls.import_dataset(filename))

    def pmf(self, x: int):
        """Return the result of the value mapped into Probability Mass Function
        of the Binomial instance.

        For a binomial distribution with n trials and probability p,
        the Probability Mass Function calculates the likelihood of getting x positive outcomes.

        :param x: a point for calculating the Probability Mass Function.
        :return: the output of Probability Mass Function.
        """

        if x < 0 or type(x) is not int:
            raise ValueError("A positive integer expected.")
        elif x > self.n:
            raise ValueError(f"The input value must be smaller than or equal {self.n}.")

        return math.comb(self.n, x) * self.p**x * self.q**(self.n - x)

    def probability(self, k: int):
        """Return the probability of getting exactly k successes in n independent Bernoulli trials.

        :param k: number of successes.
        :return: probability of k.
        """

        return self.pmf(k)

    def plot_histogram(self):
        """Plot the histogram of the dataset."""

        if len(self._data):
            plt.bar(x=['0', '1'], height=[self.__q*self.__n, self.mean])
            plt.title('Histogram')
            plt.xlabel('outcome')
            plt.ylabel('count')
            plt.show()

        else:
            raise ValueError('Load the dataset to the instance first to plot the graph.')

    def plot_pmf(self):
        """Plot the Probability Mass Function of the instance."""

        if len(self.data):
            x = []
            y = []
            for i in range(self.__n + 1):
                x.append(i)
                y.append(self.pmf(i))

            # plot the probability mass function
            plt.plot(x, y)
            plt.title('Distribution of outcomes')
            plt.xlabel('outcome')
            plt.ylabel('probability')
            plt.show()
            # return x, y

        else:
            raise ValueError('Load the dataset first to plot the graphs.')

    def __add__(self, other):
        if type(other) is Binomial:
            if self.p == other.p:
                return Binomial(self.n + other.n, self.p)
            else:
                return NotImplemented

        else:
            return NotImplemented
