import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp
import pandas as pd
from astropy.timeseries import LombScargle


# ==================================================================
# UTILS
# ==================================================================

def normalize_pdf(P):
    return P / np.sum(P)


def marginalize(P):
    P_beta = normalize_pdf(np.sum(P, axis=(1, 2)))
    P_i = normalize_pdf(np.sum(P, axis=(0, 2)))
    P_bp = normalize_pdf(np.sum(P, axis=(0, 1)))
    return P_beta, P_i, P_bp


def mean_std(grid, P):
    sum_P = np.sum(P)
    if sum_P == 0:
        return grid[len(grid) // 2], 0.0
    mean = np.sum(grid * P) / sum_P
    var = np.sum((grid - mean) ** 2 * P) / sum_P
    return mean, np.sqrt(var)


def credible_interval_hpd(grid, P, alpha=0.68):
    sum_P = np.sum(P)
    if sum_P == 0:
        return grid.min(), grid.max()

    idx = np.argsort(P)[::-1]
    P_sorted = P[idx]
    grid_sorted = grid[idx]

    cumsum = np.cumsum(P_sorted)
    cumsum /= cumsum[-1]

    cutoff_idx = np.searchsorted(cumsum, alpha)
    selected = grid_sorted[:cutoff_idx + 1]

    return selected.min(), selected.max()


def find_all_modes_deflation(P, r_beta=3, r_i=3, r_bp=35, max_modes=10, threshold_ratio=0.08):
    P_copy = P.copy()
    modes = []

    global_max_val = np.max(P_copy)
    if global_max_val == 0:
        return modes

    cutoff = global_max_val * threshold_ratio

    for _ in range(max_modes):
        idx = np.unravel_index(np.argmax(P_copy), P_copy.shape)
        val = P_copy[idx]

        if val < cutoff:
            break

        modes.append(idx)

        b_min = max(0, idx[0] - r_beta)
        b_max = min(P.shape[0], idx[0] + r_beta + 1)
        i_min = max(0, idx[1] - r_i)
        i_max = min(P.shape[1], idx[1] + r_i + 1)
        bp_min = max(0, idx[2] - r_bp)
        bp_max = min(P.shape[2], idx[2] + r_bp + 1)

        P_copy[b_min:b_max, i_min:i_max, bp_min:bp_max] = 0.0

    return modes


def extract_local(P, center, beta_vec, i_vec, bp_vec, window_spatial=3, window_bp=35):
    ib, ii, ibp = center

    b_min = max(0, ib - window_spatial)
    b_max = min(len(beta_vec), ib + window_spatial + 1)
    i_min = max(0, ii - window_spatial)
    i_max = min(len(i_vec), ii + window_spatial + 1)
    bp_min = max(0, ibp - window_bp)
    bp_max = min(len(bp_vec), ibp + window_bp + 1)

    return (
        P[b_min:b_max, i_min:i_max, bp_min:bp_max],
        beta_vec[b_min:b_max],
        i_vec[i_min:i_max],
        bp_vec[bp_min:bp_max]
    )


def analyze_and_visualize_posterior(log_posterior_map, beta_vec, i_vec, bp_vec, star_name, show_plot=True):
    logZ = logsumexp(log_posterior_map)
    posterior = np.exp(log_posterior_map - logZ)

    rad2deg = 180 / np.pi

    # Расчет всех маргинальных проекций
    P_beta, P_i, P_bp = marginalize(posterior)
    P_beta_i = np.sum(posterior, axis=2)
    P_beta_bp = np.sum(posterior, axis=1)
    P_i_bp = np.sum(posterior, axis=0)

    # Поиск локальных максимумов (с обновленными параметрами фильтрации)
    modes = find_all_modes_deflation(posterior, r_beta=4, r_i=4, r_bp=45, threshold_ratio=0.07)

    print(f"\n[{star_name}] Найдено значимых изолированных мод: {len(modes)}")

    modes_metrics = []
    for idx, mode_pos in enumerate(modes, start=1):
        post_l, beta_l, i_l, bp_l = extract_local(posterior, mode_pos, beta_vec, i_vec, bp_vec)

        mode_probability = np.sum(post_l)
        P_beta_l, P_i_l, P_bp_l = marginalize(post_l)

        b_mean, b_std = mean_std(beta_l, P_beta_l)
        i_mean, i_std = mean_std(i_l, P_i_l)
        bp_mean, bp_std = mean_std(bp_l, P_bp_l)

        b_ci_68 = credible_interval_hpd(beta_l, P_beta_l, alpha=0.68)
        b_ci_95 = credible_interval_hpd(beta_l, P_beta_l, alpha=0.95)

        i_ci_68 = credible_interval_hpd(i_l, P_i_l, alpha=0.68)
        i_ci_95 = credible_interval_hpd(i_l, P_i_l, alpha=0.95)

        bp_ci_68 = credible_interval_hpd(bp_l, P_bp_l, alpha=0.68)
        bp_ci_95 = credible_interval_hpd(bp_l, P_bp_l, alpha=0.95)

        print(f"\n===== MODE {idx} (Вероятность: {mode_probability * 100:.2f}%) =====")
        print(
            f"beta = {b_mean * rad2deg:.2f}° ± {b_std * rad2deg:.2f}° | 68% CI: [{b_ci_68[0] * rad2deg:.1f}°, {b_ci_68[1] * rad2deg:.1f}°] | 95% CI: [{b_ci_95[0] * rad2deg:.1f}°, {b_ci_95[1] * rad2deg:.1f}°]")
        print(
            f"i    = {i_mean * rad2deg:.2f}° ± {i_std * rad2deg:.2f}° | 68% CI: [{i_ci_68[0] * rad2deg:.1f}°, {i_ci_68[1] * rad2deg:.1f}°] | 95% CI: [{i_ci_95[0] * rad2deg:.1f}°, {i_ci_95[1] * rad2deg:.1f}°]")
        print(
            f"B_p  = {bp_mean:.1f} ± {bp_std:.1f} G     | 68% CI: [{bp_ci_68[0]:.1f}, {bp_ci_68[1]:.1f}] G | 95% CI: [{bp_ci_95[0]:.1f}, {bp_ci_95[1]:.1f}] G")

        modes_metrics.append({
            "mode_index": idx,
            "probability": mode_probability,
            "grid_coord": mode_pos,
            "beta": {"mean": b_mean, "std": b_std, "ci_68": b_ci_68, "ci_95": b_ci_95},
            "i": {"mean": i_mean, "std": i_std, "ci_68": i_ci_68, "ci_95": b_ci_95},
            "bp": {"mean": bp_mean, "std": bp_std, "ci_68": bp_ci_68, "ci_95": bp_ci_95}
        })

    # ==================================================================
    # ВЫЧИСЛЕНИЕ ГЛОБАЛЬНОГО МАКСИМУМА АПОСТЕРИОРНОЙ ВЕРОЯТНОСТИ (MAP)
    # ==================================================================
    global_max_idx = np.unravel_index(np.argmax(log_posterior_map), log_posterior_map.shape)
    beta_map = beta_vec[global_max_idx[0]] * rad2deg
    i_map = i_vec[global_max_idx[1]] * rad2deg
    bp_map = bp_vec[global_max_idx[2]]

    print("\n" + "=" * 60)
    print(" GLOBAL MAXIMUM A POSTERIORI (MAP) ESTIMATE:")
    print("=" * 60)
    print(f"Индексы ячейки сетки: {global_max_idx}")
    print(f"beta (MAP) = {beta_map:.2f}°")
    print(f"i    (MAP) = {i_map:.2f}°")
    print(f"B_p  (MAP) = {bp_map:.1f} G")
    print("=" * 60 + "\n")

    # Отрисовка чистого финального графика
    if show_plot:
        fig, axes = plt.subplots(3, 3, figsize=(19, 10))

        params = [beta_vec, i_vec, bp_vec]
        labels = [r"$\beta$ (rad)", r"$i$ (rad)", r"$B_p$ (G)"]
        P_1D = [P_beta, P_i, P_bp]

        # Контрастные цвета для маркеров мод на черно-белом фоне
        mode_colors = ['red', 'darkorange', 'magenta', 'cyan', 'lime', 'blue']

        # 1. Чистая диагональ (1D маргиналы)
        for i in range(3):
            ax = axes[i, i]
            ax.plot(params[i], P_1D[i], color='black', lw=1.5, zorder=2)
            ax.fill_between(params[i], P_1D[i], alpha=0.15, color='gray')
            ax.set_title(labels[i], fontsize=12)
            ax.set_yticks([])
            ax.set_xlim(params[i].min(), params[i].max())

        # 2. Внедиагональные 2D карты (Палитра заменена на черно-белую 'gray_r')
        pairs = {
            (1, 0): P_beta_i,
            (2, 0): P_beta_bp,
            (2, 1): P_i_bp,
        }

        for (i, j), P_2d in pairs.items():
            P_s = gaussian_filter(P_2d, sigma=1.0)
            # Применяем черно-белую палитру gray_r
            contour = axes[i, j].contourf(params[j], params[i], P_s.T, levels=25, cmap='gray_r')

            axes[i, j].set_xlim(params[j].min(), params[j].max())
            axes[i, j].set_ylim(params[i].min(), params[i].max())

            cbar = fig.colorbar(contour, ax=axes[i, j], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.yaxis.get_offset_text().set_fontsize(8)

            for m_idx, m_data in enumerate(modes_metrics):
                m_pos = m_data["grid_coord"]
                color = mode_colors[m_idx % len(mode_colors)]
                coords_2d = [m_pos[0], m_pos[1], m_pos[2]]

                axes[i, j].scatter(params[j][coords_2d[j]], params[i][coords_2d[i]],
                                   color=color, marker='x', s=55, lw=2.2)

        # 3. Чистая легенда
        legend_elements = []
        for m_idx, m_data in enumerate(modes_metrics):
            color = mode_colors[m_idx % len(mode_colors)]
            prob = m_data["probability"] * 100
            legend_elements.append(
                Line2D([0], [0], marker='x', color=color, linestyle='None',
                       markersize=8, markeredgewidth=2, label=f'Mode {m_idx + 1} ({prob:.1f}%)')
            )

        axes[0, 1].legend(handles=legend_elements, loc='center', fontsize=11, frameon=True, shadow=True)

        for i in range(3):
            for j in range(i + 1, 3):
                axes[i, j].axis("off")

        for j in range(3):
            axes[2, j].set_xlabel(labels[j], fontsize=10)
        for i in range(3):
            axes[i, 0].set_ylabel(labels[i], fontsize=10)

        # ==================================================================
        # СИНХРОНИЗАЦИЯ ГЕОМЕТРИИ ОСЕЙ
        # ==================================================================
        # Сначала вызываем tight_layout, чтобы зафиксировать базовые позиции
        plt.tight_layout()

        # Колонка 0: сжимаем 1D график (0,0) до точной ширины 2D графика (1,0) под ним
        pos_2d_col0 = axes[1, 0].get_position()
        pos_1d_col0 = axes[0, 0].get_position()
        axes[0, 0].set_position([pos_2d_col0.x0, pos_1d_col0.y0, pos_2d_col0.width, pos_1d_col0.height])

        # Колонка 1: сжимаем 1D график (1,1) до точной ширины 2D графика (2,1) под ним
        pos_2d_col1 = axes[2, 1].get_position()
        pos_1d_col1 = axes[1, 1].get_position()
        axes[1, 1].set_position([pos_2d_col1.x0, pos_1d_col1.y0, pos_2d_col1.width, pos_1d_col1.height])

        # Колонка 2: у 1D графика (2,2) нет соседа снизу, но для идеальной симметрии всей сетки сужаем и его
        pos_1d_col2 = axes[2, 2].get_position()
        axes[2, 2].set_position([pos_1d_col2.x0, pos_1d_col2.y0, pos_2d_col1.width, pos_1d_col2.height])
        # ==================================================================

        plot_filename = f"{star_name}_corner_plot.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"\n[График сохранен]: {plot_filename}")
        plt.show()

    return {
        "detected_modes_count": len(modes),
        "modes_list": modes_metrics,
        "marginalized_1d": {"P_beta": P_beta, "P_i": P_i, "P_bp": P_bp},
        "marginalized_2d": {"P_beta_i": P_beta_i, "P_beta_bp": P_beta_bp, "P_i_bp": P_i_bp}
    }


# ==================================================================
# MAIN PIPELINE FUNCTION
# ==================================================================

def process_star_data(star_name, observ_data='Test_synt_data.csv', python_compute=False, phase_mode=True):
    """

    :param star_name: string: Название звезды
    :param observ_data: string: путь к файлу с наблюдаемыми данными, нужен только для подхода через питон.
    :param python_compute: logical: переменная включения и отключения работы через питон
    :param phase_mode: logical: переменная включения/отключения работы в режиме с известными фазами.
    :return:
    """
    print(f"\n" + "=" * 60)
    print(f" СТАРТ ПОЛНОГО АНАЛИЗА ДЛЯ ЗВЕЗДЫ: {star_name}")
    print("=" * 60)

    # Настройка сеток для определения параметров. Без особых проблем работает только при использовании питона.
    # Для фортрана нужно отдельно сводить эти сетки. Пока что в ручном режиме. Так как в фортране, свое определение сеток.
    # Программа field_curve_modern_fortran.f95. Есть аналогичные строчки, только количество фаз чуть спрятано, но переменные названы аналогично.
    # Параметры подобраны оптимальным способом, все что связано с углами имеет шаг 5 градусов, с полем шаг 80 Гс.

    num_i = 36
    num_beta = 36
    num_bp0 = 500
    num_phases = 72

    bp = np.linspace(0, 4.0E+4, num_bp0)
    i_vector = np.linspace(0, np.pi, num_i)
    beta_vector = np.linspace(0, np.pi, num_beta)

    try:
        df = pd.read_csv(observ_data)
    except FileNotFoundError:
        df = pd.DataFrame({'phase': np.linspace(0, 1, 72), '<B_l>': np.zeros(72), '<B_err>': np.ones(72)})

    if phase_mode:
        phi_vector = df['phase'].values
    else:
        phi_vector = np.linspace(0, 1.0, num_phases)

    if python_compute:
        observe_data = df['<B_l>'].values
        observe_err = df['<B_err>'].values
        t1 = time.time()
        log_posterior_map = magnetic_model.posterior_result(
            observe_data, observe_err, i_vector, beta_vector, bp, phase_mode, phi_vector
        )
        t2 = time.time()
        print(f"Compute time: {t2 - t1:.2f} s")
    else:
        data = np.loadtxt("./Fortran_code/fortran_maps_output.dat")
        assert data.shape == (num_beta * num_i, num_bp0)
        log_posterior_map = data.reshape((num_beta, num_i, num_bp0))

    results = analyze_and_visualize_posterior(log_posterior_map, beta_vector, i_vector, bp, star_name=star_name)

    dist_1d = results["marginalized_1d"]
    dist_2d = results["marginalized_2d"]

    # --------------------------------------------------------------
    # ТОТАЛЬНОЕ СОХРАНЕНИЕ ВСЕХ КАРТ И СЕЧЕНИЙ НА ДИСК
    # --------------------------------------------------------------
    # 1. Сохранение всех 1D маргинальных распределений
    np.savetxt(f"{star_name}_marginalized_P_beta.txt", np.column_stack((beta_vector, dist_1d["P_beta"])),
               header="beta_rad Probability_Density")
    np.savetxt(f"{star_name}_marginalized_P_i.txt", np.column_stack((i_vector, dist_1d["P_i"])),
               header="i_rad Probability_Density")
    np.savetxt(f"{star_name}_marginalized_P_bp.txt", np.column_stack((bp, dist_1d["P_bp"])),
               header="B_p_G Probability_Density")

    # 2. Сохранение всех 2D матриц распределений (карт)
    np.savetxt(f"{star_name}_matrix_P_beta_i.txt", dist_2d["P_beta_i"])
    np.savetxt(f"{star_name}_matrix_P_beta_bp.txt", dist_2d["P_beta_bp"])
    np.savetxt(f"{star_name}_matrix_P_i_bp.txt", dist_2d["P_i_bp"])

    print(f"\n[Все сырые распределения успешно экспортированы]:")
    print(f" -> 1D файлы: {star_name}_marginalized_P_(beta/i/bp).txt")
    print(f" -> 2D матрицы: {star_name}_matrix_P_(beta_i/beta_bp/i_bp).txt")
    print("=" * 60 + "\n")
    return results


def compute_period_by_ls(time_series, magnetic_field_long, err_magnetic_field_long, plot=False):
    ls = LombScargle(time_series, magnetic_field_long, err_magnetic_field_long)

    # Автоматически вычисленная сетка частот и мощность периодограммы. В более менее разумных пределах.
    frequency, power = ls.autopower(
        minimum_frequency=1 / 20.0, maximum_frequency=1 / 0.2
    )

    # Максимальная частота
    best_freq = frequency[np.argmax(power)]
    best_period = 1.0 / best_freq

    # Вычисление False Alarm Probability (FAP) для максимального пика
    fap = ls.false_alarm_probability(power.max())

    print(f"Наилучший найденный период: {best_period:.5f} дн.")
    print(f"Вероятность ложной тревоги (FAP): {fap:.2e}")

    # ==========================================
    # 3. ПОСТРОЕНИЕ ГРАФИКОВ
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Левый график: Периодограмма
    ax1.plot(1.0 / frequency, power, color="black", lw=1.5)
    ax1.axvline(best_period, color="red", linestyle="--", alpha=0.7,
                label=f"P = {best_period:.3f} d")
    ax1.set_title("Периодограмма Ломба — Скаргла")
    ax1.set_xlabel("Период (дни)")
    ax1.set_ylabel("Мощность (Power)")
    ax1.set_xscale("log")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # Правый график: Свернутая фазовая кривая (магнитная кривая)
    # Вычисление фазы для каждой точки: фаза = дробная часть ( (t - t0) / P )
    t0 = time_series[0]  # начальная эпоха
    phases = np.remainder(time_series - t0, best_period) / best_period

    phases_extended = np.concatenate([phases, phases + 1.0])
    B_extended = np.concatenate([magnetic_field_long, magnetic_field_long])
    e_extended = np.concatenate([err_magnetic_field_long, err_magnetic_field_long])

    ax2.errorbar(phases_extended, B_extended, yerr=e_extended, fmt="o",
                 color="darkblue", ecolor="gray", capsize=3, elinewidth=1,
                 alpha=0.8, label="Данные")

    # Построение плавной аналитической кривой по найденному периоду
    phase_fit = np.linspace(0, 2, 200)
    t_fit = t0 + phase_fit * best_period
    B_fit = ls.model(t_fit, best_freq)
    ax2.plot(phase_fit, B_fit, color="crimson", lw=2, label="Модель")

    ax2.set_title("Магнитная кривая (свернутая по фазе)")
    ax2.set_xlabel("Фаза вращения")
    ax2.set_ylabel("Магнитное поле (Гс)")
    ax2.set_xlim(0, 2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.close()

    return best_period, fap


if __name__ == '__main__':
    star = "52her_liter"
    num_random_phase = 5

    num_of_test = 20

    dir_name_test_star = 'test_stars'

    jd0 = 2453600.975
    p0 = 3.8575

    phase_mode_test = False

    observ_data_init = pd.read_csv(f"{dir_name_test_star}/{star}", sep=';')

    t = observ_data_init['JD']
    B = observ_data_init['Bl']
    e = observ_data_init['errBl']

    p0_new, _ = compute_period_by_ls(t, B, e)

    phase_data_init = np.remainder(t - jd0, p0) / p0

    observ_data_init['phase'] = phase_data_init

    num_phases_init = len(phase_data_init)

    for i in range(num_of_test):

        rng = np.random.default_rng()

        index_phase = rng.choice(np.arange(0, num_phases_init), size=num_random_phase, replace=False)

        dir_fortran_code = f'./Fortran_code/fortran_data_{i+1}.dat'

        if phase_mode_test:
            with open(dir_fortran_code, 'w') as f:
                f.write(f'{num_random_phase}\n')
                observ_data_init.iloc[index_phase].sort_values(by='phase').to_csv(f, index=False,
                                                                                  columns=['phase', 'Bl', 'errBl'],
                                                                                  header=False, sep=' ')

            observ_data_init.iloc[index_phase].sort_values(by='phase').to_csv('Test_synt_data.csv', index=False,
                                                                              columns=['phase', 'Bl', 'errBl'])
        else:
            with open(dir_fortran_code, 'w') as f:
                f.write(f'{num_random_phase}\n')
                observ_data_init.iloc[index_phase].sort_values(by='phase').to_csv(f, index=False,
                                                                                  columns=['Bl', 'errBl'],
                                                                                  header=False, sep=' ')

            observ_data_init.iloc[index_phase].sort_values(by='phase').to_csv(f'Test_synt_data_{i+1}.csv', index=False,
                                                                              columns=['Bl', 'errBl'])

    # star_results = process_star_data(star + f'_woutphi_{num_random_phase}', python_compute=False, phase_mode=False)
