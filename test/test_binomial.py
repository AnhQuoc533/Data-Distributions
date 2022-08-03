from .. import Binomial
import pytest

binomial = Binomial(20, .4)


def test_init():
    assert binomial.p == .4
    assert binomial.n == 20
    assert binomial.q == 1 - .4
    assert len(binomial.data) == 0

    assert binomial.mean == 8
    assert round(binomial.std, 2) == 2.19


def test_invalid_init():
    with pytest.raises(ValueError):
        _ = Binomial(0, 1)

    with pytest.raises(ValueError):
        _ = Binomial(5, 1.5)

    with pytest.raises(ValueError):
        _ = Binomial(-1, 0.5)

    with pytest.raises(ValueError):
        _ = Binomial(7, -0.25)

    with pytest.raises(ValueError):
        _ = Binomial(5.5, 0.3)


def test_plot_ValueError():
    with pytest.raises(ValueError):
        binomial.plot_histogram()
    with pytest.raises(ValueError):
        binomial.plot_pmf()


def test_load_data():
    data = [True, False, False, True, False, False]
    binomial_1 = Binomial.from_binary_data(data)
    assert binomial_1.n == len(data)
    assert len(binomial_1.data) == len(data)
    assert binomial_1.data == [1, 0, 0, 1, 0, 0]
    assert binomial_1.p == 2 / 6

    assert binomial_1.mean == 2
    assert pytest.approx(binomial_1.std) == 1.1547

    data = [1., 0., 0., 1, .0, .0]
    binomial_2 = Binomial.from_binary_data(data)
    assert binomial_2.n == len(data)
    assert len(binomial_2.data) == len(data)
    assert binomial_2.data == [1, 0, 0, 1, 0, 0]
    assert binomial_2.p == 2 / 6

    binomial_1.plot_histogram()
    binomial_1.plot_pmf()


def test_invalid_data():
    with pytest.raises(ValueError):
        _ = Binomial.from_binary_data([])

    with pytest.raises(ValueError):
        _ = Binomial.from_binary_data([1, 2, 3, 4, 5])

    with pytest.raises(ValueError):
        _ = Binomial.from_binary_data(['1', '2', '3, 4, 5'])


def test_read_file():
    binomial_1 = Binomial.from_file('binomial_data.txt')
    assert len(binomial_1.data) == binomial_1.n == 13
    assert binomial_1.data == [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    assert round(binomial_1.p, 3) == .615


def test_probability():
    assert round(binomial.probability(5), 5) == .07465
    assert round(binomial.probability(3), 5) == .01235

    binomial_1 = Binomial.from_file('binomial_data.txt')
    assert round(binomial_1.probability(5), 5) == .05439
    assert round(binomial_1.probability(3), 5) == .00472

    binomial_2 = Binomial(60, .15)
    assert round(binomial_2.probability(7), 2) == .12


def test_add():
    binomial_1 = Binomial(20, .4)
    binomial_2 = Binomial(60, .4)
    binomial_sum = binomial_1 + binomial_2
    assert binomial_sum.p == .4
    assert binomial_sum.n == 80
    assert binomial_sum.data == []

    with pytest.raises(TypeError):
        _ = 1 + binomial_1

    with pytest.raises(TypeError):
        _ = binomial_1 + 8

    with pytest.raises(TypeError):
        _ = Binomial(20, .5) + binomial_2
