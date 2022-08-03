from .. import Gaussian
import pytest
gaussian_one = Gaussian(25, 3)
gaussian_two = Gaussian(30, 4)


def test_init():
    assert gaussian_one.mean == 25 and gaussian_one.std == 3
    assert str(gaussian_one) == 'Mean: 25 - Standard Deviation: 3'
    assert len(gaussian_one.data) == 0

    assert gaussian_two.mean == 30 and gaussian_two.std == 4
    assert str(gaussian_two) == 'Mean: 30 - Standard Deviation: 4'
    assert len(gaussian_two.data) == 0


def test_plot_ValueError():
    with pytest.raises(ValueError):
        gaussian_one.plot_histogram()
    with pytest.raises(ValueError):
        gaussian_one.plot_pdf()


def test_load_data():
    data = [26, 33, 65, 28, 34, 55, 25, 44, 50, 36, 26, 37, 43, 62, 35, 38, 45, 32, 28, 34]

    gaussian = Gaussian.from_dataset(data, is_sample=True)
    assert gaussian.mean == 38.8
    assert pytest.approx(gaussian.std) == 11.69615321
    assert len(gaussian.data) == len(data)

    gaussian = Gaussian.from_dataset(data, is_sample=False)
    assert gaussian.mean == 38.8
    assert pytest.approx(gaussian.std) == 11.4
    assert len(gaussian.data) == len(data)

    gaussian.plot_histogram()
    gaussian.plot_pdf()


def test_invalid_data():
    with pytest.raises(ValueError):
        _ = Gaussian.from_dataset([], True)


def test_read_file():
    gaussian = Gaussian.from_file('gaussian_data.txt', is_sample=True)
    assert gaussian.mean == sum(gaussian.data) / len(gaussian.data)
    assert pytest.approx(gaussian.std) == 92.87459776004906
    assert len(gaussian.data) == 11
    assert tuple(gaussian.data) == (1, 3, 99, 100, 120, 32, 330, 23, 76, 44, 31), 'The data file is not read correctly.'


def test_pdf():
    gaussian = Gaussian.from_file('gaussian_data.txt', is_sample=True)
    assert round(gaussian.pdf(75), 5) == 0.00429

    gaussian = Gaussian(25, 2)
    assert round(gaussian.pdf(25), 5) == 0.19947


def test_z_score():
    gaussian = Gaussian(38.8, 11.4)
    assert pytest.approx(gaussian.z_score(26)) == -1.122807018
    assert pytest.approx(gaussian.z_score(65)) == 2.298245614


def test_add():
    new_gaussian = gaussian_one + gaussian_two

    assert new_gaussian.mean == 55
    assert new_gaussian.std == 5
    assert new_gaussian.data == []


def test_TypeError():
    with pytest.raises(TypeError):
        _ = gaussian_one + 1

    with pytest.raises(TypeError):
        _ = gaussian_one - 1

    with pytest.raises(TypeError):
        _ = 1 + gaussian_one


def test_probability():
    gaussian_1 = Gaussian(180, 34)
    assert gaussian_1.probability(185, 185) == 0
    assert gaussian_1.probability() == 1
    assert pytest.approx(gaussian_1.probability(120, 155)) == 0.19227359
    assert pytest.approx(gaussian_1.probability(155, 120)) == 0.19227359

    gaussian_2 = Gaussian(30, 4)
    assert pytest.approx(gaussian_2.probability(a=28)) == 0.6914624613

    gaussian_3 = Gaussian(82, 8)
    assert round(gaussian_3.probability(b=84), 4) == 0.5987
