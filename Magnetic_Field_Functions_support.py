import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def c_1(u):
    denim = 3.0 - u
    return (15.0 + u) / (20.0 * denim)


def c_2(u):
    denim = 3.0 - u
    return 3.0 * (0.77778 - 0.22613 * u) / denim


def c_3(u):
    denim = 3.0 - u
    return 3.0 * (0.64775 - 0.23349 * u) / denim


def cos_gamma(declination_of_rotation, declination_of_magnetic_field, phase_angle):
    return np.cos(declination_of_rotation) * np.cos(declination_of_magnetic_field) + np.sin(
        declination_of_rotation) * np.sin(declination_of_magnetic_field) * np.cos(phase_angle)


def average_magnetic_field_longitudinal(polar_magnetic_field, declination_of_rotation, declination_of_magnetic_field,
                                        phase_angle, u=0.5):
    return polar_magnetic_field * c_1(u) * cos_gamma(declination_of_rotation, declination_of_magnetic_field,
                                                     phase_angle)


def average_magnetic_field(polar_magnetic_field, declination_of_rotation, declination_of_magnetic_field,
                           phase_angle, u=0.5):
    cos_g_2 = np.power(cos_gamma(declination_of_rotation, declination_of_magnetic_field, phase_angle), 2.0)
    sin_g_2 = 1.0 - cos_g_2

    return polar_magnetic_field * (c_2(u) * cos_g_2 + c_3(u) * sin_g_2)


if __name__ == '__main__':
    u_dark_parameter = 0.5
    num_models = 100

    i_rad = np.random.uniform(0.0, np.pi, num_models)
    beta_rad = np.random.uniform(0.0, np.pi, num_models)

    polar_field = np.random.uniform(1000.0, 8000.0, num_models)

    phase_array = np.sort(np.random.uniform(0, 2.0 * np.pi, 10))

    for i in range(num_models):
        magnetic_field = average_magnetic_field(polar_field[i], i_rad[i], beta_rad[i], phase_array)

        plt.plot(phase_array, magnetic_field)
    plt.show()
