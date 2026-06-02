import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp
import pandas as pd


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

    # Поиск локальных максимумов
    # Если явно находятся лишние пики, срочно увеличить threshold_ratio, по сути уровень сигнала, ниже которого искать не стоит.
    # Также аргументом существует максимум количества пиков (max_modes). Тоже можно увеличить в случае чего.
    modes = find_all_modes_deflation(posterior, r_beta=3, r_i=3, r_bp=35, threshold_ratio=0.08)

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

    global_max_idx = np.unravel_index(np.argmax(log_posterior_map), log_posterior_map.shape)
    beta_map = beta_vec[global_max_idx[0]] * rad2deg
    i_map = i_vec[global_max_idx[1]] * rad2deg
    bp_map = bp_vec[global_max_idx[2]]

    print("\n" + "=" * 50)
    print(" GLOBAL MAXIMUM A POSTERIORI (MAP) ESTIMATE:")
    print("=" * 50)
    print(f"Индексы ячейки сетки: {global_max_idx}")
    print(f"beta (MAP) = {beta_map:.2f}°")
    print(f"i    (MAP) = {i_map:.2f}°")
    print(f"B_p  (MAP) = {bp_map:.1f} G")
    print("=" * 50 + "\n")

    # Отрисовка чистого финального графика
    if show_plot:
        fig, axes = plt.subplots(3, 3, figsize=(17, 10))

        params = [beta_vec, i_vec, bp_vec]
        labels = [r"$\beta$ (rad)", r"$i$ (rad)", r"$B_p$ (G)"]
        P_1D = [P_beta, P_i, P_bp]

        mode_colors = ['red', 'darkorange', 'magenta', 'cyan', 'lime']

        # 1. Чистая диагональ (1D маргиналы без вертикальных линий)
        for i in range(3):
            ax = axes[i, i]
            ax.plot(params[i], P_1D[i], color='black', lw=1.5, zorder=2)
            ax.fill_between(params[i], P_1D[i], alpha=0.15, color='gray')
            ax.set_title(labels[i], fontsize=12)
            ax.set_yticks([])

        # 2. Внедиагональные 2D карты со шкалами
        pairs = {
            (1, 0): P_beta_i,
            (2, 0): P_beta_bp,
            (2, 1): P_i_bp,
        }

        for (i, j), P_2d in pairs.items():
            P_s = gaussian_filter(P_2d, sigma=1.0)
            contour = axes[i, j].contourf(params[j], params[i], P_s.T, levels=25, cmap='viridis')

            cbar = fig.colorbar(contour, ax=axes[i, j], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.yaxis.get_offset_text().set_fontsize(8)

            for m_idx, m_data in enumerate(modes_metrics):
                m_pos = m_data["grid_coord"]
                color = mode_colors[m_idx % len(mode_colors)]
                coords_2d = [m_pos[0], m_pos[1], m_pos[2]]

                axes[i, j].scatter(params[j][coords_2d[j]], params[i][coords_2d[i]],
                                   color=color, marker='x', s=55, lw=2.2)

        # 3. Чистая легенда (только маркеры мод и их значимость)
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

        plt.tight_layout()
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
    """"""
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


if __name__ == '__main__':
    star = "HD118022"
    star_results = process_star_data(star, python_compute=False, phase_mode=True)