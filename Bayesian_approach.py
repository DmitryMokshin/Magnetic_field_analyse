import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp
import pandas as pd

from magnetic_model import magnetic_model

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
    mean = np.sum(grid * P)
    var = np.sum((grid - mean) ** 2 * P)
    return mean, np.sqrt(var)


def credible_interval_hpd(grid, P, alpha=0.68):
    idx = np.argsort(P)[::-1]
    P_sorted = P[idx]
    grid_sorted = grid[idx]

    cumsum = np.cumsum(P_sorted)
    cumsum /= cumsum[-1]

    mask = cumsum <= alpha
    selected = grid_sorted[mask]

    return selected.min(), selected.max()


def find_two_modes(P, min_dist=5):
    flat_idx = np.argsort(P.ravel())[::-1]

    idx1 = np.unravel_index(flat_idx[0], P.shape)

    for k in flat_idx[1:]:
        idx2 = np.unravel_index(k, P.shape)
        if (
            abs(idx1[0] - idx2[0]) > min_dist
            or abs(idx1[1] - idx2[1]) > min_dist
        ):
            return idx1, idx2

    return idx1, None


def extract_local(P, center, beta_vec, i_vec, window=4):
    ib, ii, _ = center

    b_min = max(0, ib - window)
    b_max = min(len(beta_vec), ib + window + 1)

    i_min = max(0, ii - window)
    i_max = min(len(i_vec), ii + window + 1)

    return (
        P[b_min:b_max, i_min:i_max, :],
        beta_vec[b_min:b_max],
        i_vec[i_min:i_max],
    )

# ==================================================================
# MAIN
# ==================================================================

if __name__ == '__main__':

    # ==================================================================
    # SETTINGS
    # ==================================================================

    python_compute = False
    phase_mode = True

    num_i = 36
    num_beta = 36
    num_bp0 = 500

    bp = np.linspace(0, 4.0E+4, num_bp0)
    i_vector = np.linspace(0, np.pi, num_i)
    beta_vector = np.linspace(0, np.pi, num_beta)

    df = pd.read_csv('Test_synt_data.csv')

    if phase_mode:
        phi_vector = df['phase'].values
        num_phases = len(phi_vector)
    else:
        num_phases = 72
        phi_vector = np.linspace(0, 1.0, num_phases)

    # ==================================================================
    # COMPUTE / LOAD
    # ==================================================================

    if python_compute:

        observe_data = df['<B_l>'].values
        observe_err = df['<B_err>'].values

        t1 = time.time()

        log_posterior_map = magnetic_model.posterior_result(
            observe_data,
            observe_err,
            i_vector,
            beta_vector,
            bp,
            phase_mode,
            phi_vector
        )

        t2 = time.time()
        print(f"Compute time: {t2 - t1:.2f} s")

    else:
        data = np.loadtxt("./Fortran_code/fortran_maps_output.dat")
        assert data.shape == (num_beta * num_i, num_bp0)
        log_posterior_map = data.reshape((num_beta, num_i, num_bp0))

    # ==================================================================
    # NORMALIZATION
    # ==================================================================

    logZ = logsumexp(log_posterior_map)
    posterior = np.exp(log_posterior_map - logZ)

    # ==================================================================
    # FIND MODES
    # ==================================================================

    idx1, idx2 = find_two_modes(posterior)

    print("Mode 1:", idx1)
    print("Mode 2:", idx2)

    # ==================================================================
    # LOCAL ANALYSIS
    # ==================================================================

    post1, beta1, i1 = extract_local(posterior, idx1, beta_vector, i_vector)
    P_beta1, P_i1, P_bp1 = marginalize(post1)

    beta1_mean, beta1_std = mean_std(beta1, P_beta1)
    i1_mean, i1_std = mean_std(i1, P_i1)
    bp1_mean, bp1_std = mean_std(bp, P_bp1)

    post2, beta2, i2 = extract_local(posterior, idx2, beta_vector, i_vector)
    P_beta2, P_i2, P_bp2 = marginalize(post2)

    beta2_mean, beta2_std = mean_std(beta2, P_beta2)
    i2_mean, i2_std = mean_std(i2, P_i2)
    bp2_mean, bp2_std = mean_std(bp, P_bp2)

    # ==================================================================
    # OUTPUT MODES
    # ==================================================================

    print("\n===== MODE 1 =====")
    print(f"beta = {beta1_mean*180/np.pi:.4f} ± {beta1_std*180/np.pi:.4f}")
    print(f"i    = {i1_mean*180/np.pi:.4f} ± {i1_std*180/np.pi:.4f}")
    print(f"B_p  = {bp1_mean:.4f} ± {bp1_std:.4f}")

    print("\n===== MODE 2 =====")
    print(f"beta = {beta2_mean*180/np.pi:.4f} ± {beta2_std*180/np.pi:.4f}")
    print(f"i    = {i2_mean*180/np.pi:.4f} ± {i2_std*180/np.pi:.4f}")
    print(f"B_p  = {bp2_mean:.4f} ± {bp2_std:.4f}")

    # ==================================================================
    # GLOBAL ANALYSIS
    # ==================================================================

    P_beta, P_i, P_bp = marginalize(posterior)

    beta_mean, beta_std = mean_std(beta_vector, P_beta)
    i_mean, i_std = mean_std(i_vector, P_i)
    bp_mean, bp_std = mean_std(bp, P_bp)

    beta_ci = credible_interval_hpd(beta_vector, P_beta)
    i_ci = credible_interval_hpd(i_vector, P_i)
    bp_ci = credible_interval_hpd(bp, P_bp)

    # ==================================================================
    # MAP
    # ==================================================================

    idx = np.unravel_index(np.argmax(posterior), posterior.shape)

    beta_map = beta_vector[idx[0]]
    i_map = i_vector[idx[1]]
    bp_map = bp[idx[2]]

    print("\n===== GLOBAL =====")
    print(f"beta = {beta_mean*180/np.pi:.4f} ± {beta_std*180/np.pi:.4f}")
    print(f"i    = {i_mean*180/np.pi:.4f} ± {i_std*180/np.pi:.4f}")
    print(f"B_p  = {bp_mean:.4f} ± {bp_std:.4f}")

    print("\n===== 68% CI =====")
    print("beta:", beta_ci)
    print("i   :", i_ci)
    print("B_p :", bp_ci)

    print("\n===== MAP =====")
    print(f"beta = {beta_map*180/np.pi:.4f}")
    print(f"i    = {i_map*180/np.pi:.4f}")
    print(f"B_p  = {bp_map:.4f}")

    # ==================================================================
    # PLOTS
    # ==================================================================

    P_beta_i = np.sum(posterior, axis=2)
    P_beta_bp = np.sum(posterior, axis=1)
    P_i_bp = np.sum(posterior, axis=0)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    params = [beta_vector, i_vector, bp]
    labels = [r"$\beta$", r"$i$", r"$B_p$"]
    P_1D = [P_beta, P_i, P_bp]

    for i in range(3):
        axes[i, i].plot(params[i], P_1D[i])
        axes[i, i].fill_between(params[i], P_1D[i], alpha=0.3)
        axes[i, i].set_title(labels[i])
        axes[i, i].set_yticks([])

    pairs = {
        (1, 0): P_beta_i,
        (2, 0): P_beta_bp,
        (2, 1): P_i_bp,
    }

    for (i, j), P in pairs.items():
        P_s = gaussian_filter(P, sigma=1.0)
        axes[i, j].contourf(params[j], params[i], P_s.T, levels=30)

    for i in range(3):
        for j in range(i + 1, 3):
            axes[i, j].axis("off")

    for j in range(3):
        axes[2, j].set_xlabel(labels[j])

    for i in range(3):
        axes[i, 0].set_ylabel(labels[i])

    plt.tight_layout()
    plt.show()
