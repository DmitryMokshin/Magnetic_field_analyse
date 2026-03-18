import time

from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from Magnetic_Field_Functions_support import longitudinal_magnetic_field_landstreet


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

    return 1.0 / (declination_of_magnetic_field_max - declination_of_magnetic_field_min) * np.ones_like(
        declination_of_magnetic_field)


def prior_phase(phase):
    """
    Классическое равномерное распределение. Так как нет никаких оснований на существование выделенной фазы, на которой проходят наблюдения.
    От 0 до 360 градусов. (только программа работает со значениями от 0 до 1, хотя в целом не имеет значения)
    :param phase: float, numpy.array. In radians.
    :return: distribution: float, numpy.array.
    """

    phase_max = 1.0
    phase_min = 0.0

    return 1.0 / (phase_max - phase_min) * np.ones_like(phase)


def prior_scale_coef(b):
    """
    Априорное распределение масштабного множителя для дисперсий ошибок, учесть, что полученные ошибки принадлежат скорее методу.
    Учесть возможное реальное отклонение плюс доп. шумы. Значение от 0.1 до 2.0
    :param b: float, numpy.array.
    :return: distribution: float, numpy.array.
    """

    b_min = 0.1
    b_max = 2.0

    return 1.0 / b / np.log(b_max / b_min)


def likelihood_mod(observe_data, observe_err, model_data, num_observ, b):
    """
    Функция правдоподобия немного модернизированная с учетом масштабного множителя. Если взять его равным 1, получится классическая функция правдоподобия.
    :param observe_data: float, numpy.array observe data.
    :param observe_err: float, numpy.array observe error.
    :param model_data: float, numpy.array model data.
    :param b: float, scale factor.
    :param num_observ: integer, number of observation.
    :return: float, value likelihood.
    """

    return np.power(2.0 * np.pi, -num_observ / 2.0) * np.power(b, num_observ / 2.0) * np.power(np.prod(observe_err),
                                                                                               -1.0) * np.exp(
        -b / 2.0 * np.sum(np.power((observe_data - model_data) / observe_err, 2.0)))


def prior_combine(declines_rotation, declines_magnetic_field, polar_fields, scale_factors, phases):
    """
    Высчитывает априорное распределение всех параметров модели. На вход могут быть поданы, как векторы, так и скаляры.
    :param declines_rotation: float, numpy.array in radians.
    :param declines_magnetic_field: float, numpy.array in radians.
    :param polar_fields: float, numpy.array polar magnetic field in Gauss.
    :param scale_factors: float, numpy.array scale factor for error observe.
    :param phases: float, numpy.array phase of rotation in [0: 1].
    :return: numpy.array, distribution
    """

    num_phases = len(phases)
    num_i = len(declines_rotation)
    num_beta = len(declines_magnetic_field)
    num_fields = len(polar_fields)
    num_factors = len(scale_factors)

    result_distribution = np.zeros([num_phases, num_beta, num_i, num_fields, num_factors])

    for i in tqdm(range(num_i)):
        for j in range(num_beta):
            for k in range(num_fields):
                for l in range(num_factors):
                    result_distribution[:, j, i, k, l] = prior_scale_coef(b=scale_factors[l]) * prior_phase(
                        phase=phases) * prior_polar_magnetic_field(
                        polar_field=polar_fields[k]) * prior_declination_of_magnetic_field(
                        declination_of_magnetic_field=declines_magnetic_field[j]) * prior_declination_of_rotation(
                        declination_of_rotation=declines_rotation[i])

    return result_distribution


def posterior_compute_orig_value_without_evidence(decline_rotation, decline_magnetic_field, polar_field, scale_factor,
                                                  phases,
                                                  observe_long_field, observe_long_field_err):
    """
    Расчет апостериорного распределения, пока чистый апостериор без маргинализации. Расчет в зависимости от всех параметров модели в том числе и фазы вращения звезды.
    :param decline_rotation: float, in radians.
    :param decline_magnetic_field: float, in radians.
    :param polar_field: float, polar magnetic field in Gauss.
    :param scale_factor: scale factor for error observe.
    :param phases: float, numpy.array phase of rotation in [0: 1].
    :param observe_long_field: float, numpy.array observe longitudinal magnetic field in Gauss.
    :param observe_long_field_err: float, numpy.array observe error longitudinal magnetic field in Gauss.
    :return: float, value posterior
    """

    quad_pole = 0.0
    octo_pole = 0.0
    asim_dipole = 0.0

    num_observe = len(observe_long_field)

    model_long_field = longitudinal_magnetic_field_landstreet(phase_angles=phases,
                                                              declination_of_rotation=decline_rotation,
                                                              declination_of_magnetic_field=decline_magnetic_field,
                                                              polar_magnetic_field=polar_field,
                                                              asymptotic_dipole=asim_dipole, quad_pole=quad_pole,
                                                              octo_pole=octo_pole)

    prior_prod_value = prior_scale_coef(b=scale_factor) * prior_phase(phase=phases) * prior_polar_magnetic_field(
        polar_field=polar_field) * prior_declination_of_magnetic_field(
        declination_of_magnetic_field=decline_magnetic_field) * prior_declination_of_rotation(
        declination_of_rotation=decline_rotation)

    likelihood_value = likelihood_mod(observe_data=observe_long_field, observe_err=observe_long_field_err,
                                      model_data=model_long_field, num_observ=num_observe, b=scale_factor)

    return prior_prod_value * likelihood_value

def get_credible_levels(P, levels=[0.68, 0.95]):
    """
    Возвращает значения плотности, соответствующие заданным
    доверительным уровням.
    """
    P_flat = P.flatten()
    idx = np.argsort(P_flat)[::-1]
    P_sorted = P_flat[idx]

    cumsum = np.cumsum(P_sorted)
    cumsum /= cumsum[-1]

    values = []
    for lvl in levels:
        values.append(P_sorted[np.searchsorted(cumsum, lvl)])
    return values


if __name__ == '__main__':
    df = pd.read_csv('Test_synt_data.csv')

    bp = np.linspace(0, 1.0E+4, 250)

    i_vector = np.linspace(0, np.pi, 37)
    beta_vector = np.arange(0, np.pi, 5 * np.pi / 180.0)
    phi_vector = np.arange(0, 2.0 * np.pi, 5.0 * np.pi / 180.0)

    b_vector = np.linspace(0.1, 2.0, 40)

    t_0 = time.time()

    prior = prior_combine(
        declines_rotation=i_vector,
        declines_magnetic_field=beta_vector,
        polar_fields=bp,
        scale_factors=b_vector,
        phases=phi_vector
    )

    P_phi = np.sum(prior, axis=(1, 2, 3, 4))
    P_beta = np.sum(prior, axis=(0, 2, 3, 4))
    P_i = np.sum(prior, axis=(0, 1, 3, 4))
    P_bp = np.sum(prior, axis=(0, 1, 2, 4))
    P_b = np.sum(prior, axis=(0, 1, 2, 3))

    P_phi_beta = np.sum(prior, axis=(2, 3, 4))
    P_phi_i = np.sum(prior, axis=(1, 3, 4))
    P_phi_bp = np.sum(prior, axis=(1, 2, 4))
    P_phi_b = np.sum(prior, axis=(1, 2, 3))

    P_beta_i = np.sum(prior, axis=(0, 3, 4))
    P_beta_bp = np.sum(prior, axis=(0, 2, 4))
    P_beta_b = np.sum(prior, axis=(0, 2, 3))

    P_i_bp = np.sum(prior, axis=(0, 1, 4))
    P_i_b = np.sum(prior, axis=(0, 1, 3))

    P_bp_b = np.sum(prior, axis=(0, 1, 2))

    params = [phi_vector, beta_vector, i_vector, bp, b_vector]
    labels = [r"$\phi$", r"$\beta$", r"$i$", r"$B_p$", r"$b$"]

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))

    # === 1D распределения (диагональ) ===
    P_1D = [P_phi, P_beta, P_i, P_bp, P_b]

    for i in range(5):
        axes[i, i].plot(params[i], P_1D[i])
        axes[i, i].set_yticks([])
        axes[i, i].set_title(labels[i], fontsize=12)

    # === 2D распределения (нижний треугольник) ===
    P_2D = {
        (1, 0): P_phi_beta,
        (2, 0): P_phi_i,
        (3, 0): P_phi_bp,
        (4, 0): P_phi_b,
        (2, 1): P_beta_i,
        (3, 1): P_beta_bp,
        (4, 1): P_beta_b,
        (3, 2): P_i_bp,
        (4, 2): P_i_b,
        (4, 3): P_bp_b
    }

    for (i, j), P in P_2D.items():

        P_smooth = gaussian_filter(P, sigma=1.0)

        lvl_68, lvl_95 = get_credible_levels(P_smooth)

        levels = np.array([lvl_95, lvl_68])
        levels = np.sort(levels)
        levels = np.unique(levels)

        # заливка (красная)
        axes[i, j].contourf(
            params[j],
            params[i],
            P_smooth.T,
            levels=30,
            cmap="Reds"
        )

        # контуры
        if len(levels) >= 2:
            axes[i, j].contour(
                params[j],
                params[i],
                P_smooth.T,
                levels=levels,
                colors="black",
                linewidths=1.2
            )

    # === скрываем верхний треугольник ===
    for i in range(5):
        axes[i, i].plot(params[i], P_1D[i], color="black")
        axes[i, i].fill_between(params[i], P_1D[i], color="red", alpha=0.3)
        axes[i, i].set_yticks([])
        axes[i, i].set_title(labels[i], fontsize=12)  # ← это ключ

    # === подписи осей ===

    for i in range(5):
        for j in range(i + 1, 5):
            axes[i, j].axis("off")

    # нижняя строка → подписи X
    for j in range(5):
        axes[4, j].set_xlabel(labels[j])

    # левая колонка → подписи Y
    for i in range(5):
        axes[i, 0].set_ylabel(labels[i])

    # === убираем лишние тики внутри ===
    for i in range(5):
        for j in range(5):
            if i != 4:
                axes[i, j].set_xticklabels([])
            if j != 0:
                axes[i, j].set_yticklabels([])

    plt.tight_layout()
    plt.show()

    print(f'Time compute and plotting: {time.time() - t_0: .2f} c')