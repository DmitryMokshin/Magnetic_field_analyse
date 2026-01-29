import matplotlib.pyplot as plt

from Magnetic_Field_Functions_support import average_magnetic_field, average_magnetic_field_longitudinal
import numpy as np
import pandas as pd

def create_one_random_star(polar_magnetic_field, level_error_magnetic_field, num_phases, random_phase=False, need_zero_phase=False):
    star_i_rad = np.random.uniform(0, np.pi)
    star_beta_rad = np.random.uniform(0, np.pi)

    if need_zero_phase:
        phase_zero_rad = np.random.uniform(0, 2.0 * np.pi)
    else:
        phase_zero_rad = 0.0

    if random_phase:
        phase_rad = np.random.uniform(0, 2.0 * np.pi, num_phases) + phase_zero_rad
    else:
        phase_rad = np.linspace(0, 2.0 * np.pi, num_phases) + phase_zero_rad

    average_b_l = average_magnetic_field_longitudinal(polar_magnetic_field, star_i_rad, star_beta_rad, phase_rad)
    err_average_b_l = np.random.normal(np.abs(average_b_l) * level_error_magnetic_field, np.abs(average_b_l) * np.power(level_error_magnetic_field, 2.0))

    average_b = average_magnetic_field(polar_magnetic_field, star_i_rad, star_beta_rad, phase_rad)
    err_average_b = np.random.normal(np.abs(average_b) * level_error_magnetic_field, np.abs(average_b) * np.power(level_error_magnetic_field, 2.0))

    return pd.DataFrame({'phase': phase_rad, 'b_l': average_b_l, 'err_b_l': err_average_b_l, 'b_r': err_average_b, 'err_b': err_average_b})

if __name__ == '__main__':
    observ_data = create_one_random_star(1000.0, 0.1, 100, True, True)

    plt.errorbar(x=observ_data['phase'], y=observ_data['b_l'], yerr=observ_data['err_b_l'], fmt='o')
    plt.show()
