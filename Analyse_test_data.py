import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Bayesian_approach import process_star_data


# =================================================================
# ФУНКЦИЯ ДЛЯ ОТРИСОВКИ (чтобы не дублировать код для двух картинок)
# =================================================================
def plot_mode_results(mode_num, bp_data, i_data, beta_data, marker, color_theme):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    fig.suptitle(f'Анализ стабильности алгоритма ({star_name}) — МОДА {mode_num}',
                 fontsize=15, fontweight='bold')

    plots_config = [
        {
            'ax': axes[0], 'means': bp_data['means'], 'errs': bp_data['errs'],
            'best_mean': best_model_bp_mean, 'best_err': best_model_err_bp_mean,
            'ylabel': '$B_p$ [kG]', 'color': color_theme['bp']
        },
        {
            'ax': axes[1], 'means': i_data['means'], 'errs': i_data['errs'],
            'best_mean': best_model_i_mean, 'best_err': best_model_err_i_mean,
            'ylabel': 'Inclination, $i$ [deg]', 'color': color_theme['i']
        },
        {
            'ax': axes[2], 'means': beta_data['means'], 'errs': beta_data['errs'],
            'best_mean': best_model_beta_mean, 'best_err': best_model_err_beta_mean,
            'ylabel': 'Obliquity, $\\beta$ [deg]', 'color': color_theme['beta']
        }
    ]

    for config in plots_config:
        ax = config['ax']

        # Линия и зона погрешности эталона (Best Model)
        ax.axhline(config['best_mean'], color='red', linestyle='--', linewidth=1.8, label='Best model')
        ax.fill_between(models,
                        config['best_mean'] - config['best_err'],
                        config['best_mean'] + config['best_err'],
                        color='red', alpha=0.12)

        # Точки вычисленных значений моды
        ax.errorbar(models, config['means'], yerr=config['errs'],
                    fmt=marker, color=config['color'], ecolor='gray',
                    elinewidth=1.2, capsize=2.5, markersize=5.5, alpha=0.85,
                    label=f'Мода {mode_num} (mean $\\pm$ std)')

        ax.set_ylabel(config['ylabel'], fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Номер теста (Model ID)', fontsize=11)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)  # Корректный отступ сверху против наложения title


if __name__ == '__main__':

    # ==============================================================
    # Определение эталонных значений и названия звезды, а также параметров теста
    # ==============================================================

    star_name = 'hd34736'
    num_of_test = 100
    num_random_phase = 5

    dir_data = f'./test_star_result_{star_name}/result_posterior_maps/'

    # Эталонные значения (лучшие модели) в градусах и кГс
    best_model_bp_mean = 18.9
    best_model_err_bp_mean = 0.8

    best_model_i_mean = 68
    best_model_err_i_mean = 7

    best_model_beta_mean = 83
    best_model_err_beta_mean = 2

    models = list(range(1, num_of_test + 1))

    # ==============================================================
    # Обработка тестов
    # ==============================================================

    # Списки для первой (лучшей) моды
    bp_1_means, bp_1_errs = [], []
    i_1_means, i_1_errs = [], []
    beta_1_means, beta_1_errs = [], []

    # Списки для второй моды
    bp_2_means, bp_2_errs = [], []
    i_2_means, i_2_errs = [], []
    beta_2_means, beta_2_errs = [], []

    # Сбор данных
    for i in range(num_of_test):
        star_results = process_star_data(
            star_name + f'_woutphi_{num_random_phase}_{i + 1}',
            fortran_out_name=dir_data + f'fortran_maps_output_{i + 1}.dat',
            python_compute=False,
            phase_mode=False
        )

        data_for_extract = star_results['modes_list']
        # Сортируем моды по вероятности (от большей к меньшей)
        sorted_data = sorted(data_for_extract, key=lambda x: x["probability"], reverse=True)

        # --- ПЕРВАЯ МОДА ---
        mode_1 = sorted_data[0]
        bp_1_means.append(mode_1['bp']['mean'] / 1000)
        bp_1_errs.append(mode_1['bp']['std'] / 1000)

        # Переводим радианы в градусы для i и beta
        i_1_means.append(np.degrees(mode_1['i']['mean']))
        i_1_errs.append(np.degrees(mode_1['i']['std']))
        beta_1_means.append(np.degrees(mode_1['beta']['mean']))
        beta_1_errs.append(np.degrees(mode_1['beta']['std']))

        # --- ВТОРАЯ МОДА ---
        if len(sorted_data) > 1:
            mode_2 = sorted_data[1]
            bp_2_means.append(mode_2['bp']['mean'] / 1000)
            bp_2_errs.append(mode_2['bp']['std'] / 1000)

            # Переводим радианы в градусы
            i_2_means.append(np.degrees(mode_2['i']['mean']))
            i_2_errs.append(np.degrees(mode_2['i']['std']))
            beta_2_means.append(np.degrees(mode_2['beta']['mean']))
            beta_2_errs.append(np.degrees(mode_2['beta']['std']))
        else:
            bp_2_means.append(np.nan)
            bp_2_errs.append(np.nan)
            i_2_means.append(np.nan)
            i_2_errs.append(np.nan)
            beta_2_means.append(np.nan)
            beta_2_errs.append(np.nan)

    # =================================================================
    # ПОСТРОЕНИЕ ДВУХ ОТДЕЛЬНЫХ КАРТИНОК
    # =================================================================

    print('Средние данные тестов и отклонения:')
    print(f'Среднее дипольное поле мода 1 {round(np.mean(bp_1_means), 1)} +- {round(np.std(bp_1_means), 1)}')
    print(f'Среднее дипольное поле мода 2 {round(np.mean(bp_2_means), 1)} +- {round(np.std(bp_2_means), 1)}')

    print('=' * 30)
    print(f'Среднее дипольное поле мода 1 {round(np.mean(i_1_means), 1)} +- {round(np.std(i_1_means), 1)}')
    print(f'Среднее дипольное поле мода 2 {round(np.mean(i_2_means), 1)} +- {round(np.std(i_2_means), 1)}')

    print('=' * 30)
    print(f'Среднее дипольное поле мода 1 {round(np.mean(beta_1_means), 1)} +- {round(np.std(beta_1_means), 1)}')
    print(f'Среднее дипольное поле мода 2 {round(np.mean(beta_2_means), 1)} +- {round(np.std(beta_2_means), 1)}')


    theme_1 = {'bp': '#1f77b4', 'i': '#2ca02c', 'beta': '#ff7f0e'}
    theme_2 = {'bp': '#4b0082', 'i': '#8b0000', 'beta': '#2f4f4f'}


    plot_mode_results(
        mode_num=1,
        bp_data={'means': bp_1_means, 'errs': bp_1_errs},
        i_data={'means': i_1_means, 'errs': i_1_errs},
        beta_data={'means': beta_1_means, 'errs': beta_1_errs},
        marker='o',
        color_theme=theme_1
    )


    plot_mode_results(
        mode_num=2,
        bp_data={'means': bp_2_means, 'errs': bp_2_errs},
        i_data={'means': i_2_means, 'errs': i_2_errs},
        beta_data={'means': beta_2_means, 'errs': beta_2_errs},
        marker='^',
        color_theme=theme_2
    )

    plt.figure(1)
    plt.savefig(f'stability_analysis_{star_name}_mode_1.png', dpi=300, bbox_inches='tight')

    plt.figure(2)
    plt.savefig(f'stability_analysis_{star_name}_mode_2.png', dpi=300, bbox_inches='tight')

    plt.show()
