import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# =========================================
# Априоры
# =========================================

def prior_polar_magnetic_field(polar_field):
    """
    Априор полярного поля для дипольного случая, на основе Jeffreys prior. Минимальное поле выбирается как 25 Гс.
    Переходное значение. Максимальное значение поля 10^4, что в целом логично.
    :param polar_field: float, numpy.array.
    :return: distribution: float, numpy.array.
    """
    a = 25.0
    polar_magnetic_field_max = 1.0E+4

    return 1.0 / (polar_field + a) / np.log((a + polar_magnetic_field_max) / a)


def prior_declination_of_rotation(declination_of_rotation):
    """
    Обычное классическое распределение для угла наклона вращения
    :param declination_of_rotation: float, numpy.array. In radians
    :return: distribution: float, numpy.array.
    """

    return 0.5 * np.sin(declination_of_rotation)


def prior_declination_of_magnetic_field(declination_of_magnetic_field):
    """
    Классическое равномерное распределение (Почти нет никаких априорных знаний насчет углов). От 0 до 180 градусов.
    (только принимаются значения в радианах, хотя в целом не имеет значения)
    :param declination_of_magnetic_field: float, numpy.array. In radians.
    :return: distribution: float, numpy.array.
    """

    declination_of_magnetic_field_max = np.pi
    declination_of_magnetic_field_min = 0.0

    return 1.0 / (declination_of_magnetic_field_max - declination_of_magnetic_field_min)


def prior_phase(phase):
    """
    Классическое равномерное распределение. Так как нет никаких оснований на существование выделенной фазы, на которой проходят наблюдения.
    От 0 до 360 градусов. (только программа работает со значениями от 0 до 1, хотя в целом не имеет значения)
    :param phase: float, numpy.array. In radians.
    :return: distribution: float, numpy.array.
    """

    phase_max = 1.0
    phase_min = 0.0

    return 1.0 / (phase_max - phase_min)


if __name__ == '__main__':
    df = pd.read_csv('Test_synt_data.csv')

    mf = np.arange(0.0, 2.0E+4, 0.01)

    plt.plot(mf, prior_polar_magnetic_field(mf))
    plt.show()
