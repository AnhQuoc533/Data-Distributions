from .binomial import Binomial


class Bernoulli(Binomial):

    def __init__(self, prob: float):
        """Bernoulli distribution class for calculating and visualizing a Bernoulli distribution.

        The Bernoulli distribution is a special case of the binomial distribution with n=1.

        :param prob: the success probability.
        """

        super().__init__(1, prob)

    @classmethod
    def from_binary_data(cls, dataset):
        """Not available."""
        ...

    @classmethod
    def from_file(cls, filename: str):
        """Not available."""
        ...

    def pmf(self, k: int):
        """Return the probability p if k = 1 or the probability q if k = 0.

        :param k: 0 (failure) or 1 (success).
        :return: the output of Probability Mass Function.
        """

        if k == 1:
            return self.p
        elif k == 0:
            return self.q
        else:
            raise ValueError(f"The input value must be 0 (failure) or 1 (success).")

    def probability(self, is_success=True):
        """Return the probability for the success/failure.

        :param is_success: whether the trial is a success or failure.
        :return: probability of the success/failure.
        """

        return self.pmf(int(is_success))

    def plot_histogram(self):
        """Not available."""
        ...

    def __add__(self, other):
        return NotImplemented
