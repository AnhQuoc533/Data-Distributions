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
        """Return a binomial instance built from the input binary dataset (contains only 1 and 0).

        :param dataset: an 1D array-like binary dataset.
        :return: a new instance of Binomial class
        """

        if len(dataset):
            dataset = np.asarray(dataset, dtype=int).flatten()
        else:
            raise ValueError("The input dataset should have at least one element.")

        if set(dataset) == {0, 1}:
            instance = Binomial(len(dataset), len(dataset[dataset == 1]) / len(dataset))
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

        dataset = []
        with open(filename) as file:
            line = file.readline()
            while line:
                dataset.append(int(line))
                line = file.readline()

        return cls.from_binary_data(dataset)

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

    def __add__(self, other):
        if type(other) is Binomial:
            if self.p == other.p:
                return Binomial(self.n + other.n, self.p)
            else:
                return NotImplemented

        else:
            return NotImplemented
