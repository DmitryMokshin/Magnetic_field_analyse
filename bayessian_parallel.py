import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp
import pandas as pd
import os
import glob
import multiprocessing as mp

from magnetic_model import magnetic_model
from Bayesian_approach import find_two_modes, marginalize, extract_local, mean_std, credible_interval_hpd

# ==================================================================
# SETTINGS (глобально)
# ==================================================================

python_compute = True
phase_mode = True

num_i = 36
num_beta = 36
num_bp0 = 500

bp = np.linspace(0, 4.0E+4, num_bp0)
i_vector = np.linspace(0, np.pi, num_i)
beta_vector = np.linspace(0, np.pi, num_beta)

def process_one_file(filepath):

    base = os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs("results", exist_ok=True)

    print(f"Processing: {filepath}")

    df = pd.read_csv(filepath)

    if phase_mode:
        phi_vector = df['phase'].values
        num_phases = len(phi_vector)
    else:
        num_phases = 72
        phi_vector = np.linspace(0, 1.0, num_phases)

    # ==============================================================
    # COMPUTE / LOAD
    # ==============================================================

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
        print(f"{base}: compute time {t2 - t1:.2f} s")

    else:
        data = np.loadtxt("./Fortran_code/fortran_maps_output.dat")
        log_posterior_map = data.reshape((num_beta, num_i, num_bp0))

    # ==============================================================
    # NORMALIZATION
    # ==============================================================

    logZ = logsumexp(log_posterior_map)
    posterior = np.exp(log_posterior_map - logZ)

    # ==============================================================
    # MODES
    # ==============================================================

    idx1, idx2 = find_two_modes(posterior)

    post1, beta1, i1 = extract_local(posterior, idx1, beta_vector, i_vector)
    P_beta1, P_i1, P_bp1 = marginalize(post1)

    beta1_mean, beta1_std = mean_std(beta1, P_beta1)
    i1_mean, i1_std = mean_std(i1, P_i1)
    bp1_mean, bp1_std = mean_std(bp, P_bp1)

    if idx2 is not None:
        post2, beta2, i2 = extract_local(posterior, idx2, beta_vector, i_vector)
        P_beta2, P_i2, P_bp2 = marginalize(post2)

        beta2_mean, beta2_std = mean_std(beta2, P_beta2)
        i2_mean, i2_std = mean_std(i2, P_i2)
        bp2_mean, bp2_std = mean_std(bp, P_bp2)
    else:
        beta2_mean = i2_mean = bp2_mean = None

    # ==============================================================
    # GLOBAL
    # ==============================================================

    P_beta, P_i, P_bp = marginalize(posterior)

    beta_mean, beta_std = mean_std(beta_vector, P_beta)
    i_mean, i_std = mean_std(i_vector, P_i)
    bp_mean, bp_std = mean_std(bp, P_bp)

    beta_ci = credible_interval_hpd(beta_vector, P_beta)
    i_ci = credible_interval_hpd(i_vector, P_i)
    bp_ci = credible_interval_hpd(bp, P_bp)

    idx = np.unravel_index(np.argmax(posterior), posterior.shape)

    beta_map = beta_vector[idx[0]]
    i_map = i_vector[idx[1]]
    bp_map = bp[idx[2]]

    # ==============================================================
    # SAVE TEXT
    # ==============================================================

    out_txt = os.path.join("results", base + ".txt")

    with open(out_txt, "w") as f:
        f.write(f"{base}\n")

        # ==========================================================
        # MAP
        # ==========================================================

        f.write("===== MAP ESTIMATE =====\n")
        f.write(f"beta = {beta_map * 180 / np.pi:.6f} deg\n")
        f.write(f"i    = {i_map * 180 / np.pi:.6f} deg\n")
        f.write(f"B_p  = {bp_map:.6f}\n\n")

        # ==========================================================
        # GLOBAL POSTERIOR
        # ==========================================================

        f.write("===== GLOBAL POSTERIOR =====\n")
        f.write("Mean ± Std:\n")
        f.write(f"beta = {beta_mean * 180 / np.pi:.6f} ± {beta_std * 180 / np.pi:.6f} deg\n")
        f.write(f"i    = {i_mean * 180 / np.pi:.6f} ± {i_std * 180 / np.pi:.6f} deg\n")
        f.write(f"B_p  = {bp_mean:.6f} ± {bp_std:.6f}\n\n")

        f.write("68% HPD intervals:\n")
        f.write(f"beta: [{beta_ci[0] * 180 / np.pi:.6f}, {beta_ci[1] * 180 / np.pi:.6f}] deg\n")
        f.write(f"i   : [{i_ci[0] * 180 / np.pi:.6f}, {i_ci[1] * 180 / np.pi:.6f}] deg\n")
        f.write(f"B_p : [{bp_ci[0]:.6f}, {bp_ci[1]:.6f}]\n\n")

        # ==========================================================
        # MODE 1
        # ==========================================================

        f.write("===== MODE 1 =====\n")
        f.write(f"Index: {idx1}\n")

        f.write("Mean ± Std:\n")
        f.write(f"beta = {beta1_mean * 180 / np.pi:.6f} ± {beta1_std * 180 / np.pi:.6f} deg\n")
        f.write(f"i    = {i1_mean * 180 / np.pi:.6f} ± {i1_std * 180 / np.pi:.6f} deg\n")
        f.write(f"B_p  = {bp1_mean:.6f} ± {bp1_std:.6f}\n\n")

        # локальные интервалы
        beta1_ci = credible_interval_hpd(beta1, P_beta1)
        i1_ci = credible_interval_hpd(i1, P_i1)
        bp1_ci = credible_interval_hpd(bp, P_bp1)

        f.write("68% HPD (local):\n")
        f.write(f"beta: [{beta1_ci[0] * 180 / np.pi:.6f}, {beta1_ci[1] * 180 / np.pi:.6f}] deg\n")
        f.write(f"i   : [{i1_ci[0] * 180 / np.pi:.6f}, {i1_ci[1] * 180 / np.pi:.6f}] deg\n")
        f.write(f"B_p : [{bp1_ci[0]:.6f}, {bp1_ci[1]:.6f}]\n\n")

        # ==========================================================
        # MODE 2
        # ==========================================================

        if idx2 is not None:
            f.write("===== MODE 2 =====\n")
            f.write(f"Index: {idx2}\n")

            f.write("Mean ± Std:\n")
            f.write(f"beta = {beta2_mean * 180 / np.pi:.6f} ± {beta2_std * 180 / np.pi:.6f} deg\n")
            f.write(f"i    = {i2_mean * 180 / np.pi:.6f} ± {i2_std * 180 / np.pi:.6f} deg\n")
            f.write(f"B_p  = {bp2_mean:.6f} ± {bp2_std:.6f}\n\n")

            beta2_ci = credible_interval_hpd(beta2, P_beta2)
            i2_ci = credible_interval_hpd(i2, P_i2)
            bp2_ci = credible_interval_hpd(bp, P_bp2)

            f.write("68% HPD (local):\n")
            f.write(f"beta: [{beta2_ci[0] * 180 / np.pi:.6f}, {beta2_ci[1] * 180 / np.pi:.6f}] deg\n")
            f.write(f"i   : [{i2_ci[0] * 180 / np.pi:.6f}, {i2_ci[1] * 180 / np.pi:.6f}] deg\n")
            f.write(f"B_p : [{bp2_ci[0]:.6f}, {bp2_ci[1]:.6f}]\n\n")

            # сравнение мод
            f.write("===== MODE COMPARISON =====\n")
            f.write(f"Δbeta = {abs(beta1_mean - beta2_mean) * 180 / np.pi:.6f} deg\n")
            f.write(f"Δi    = {abs(i1_mean - i2_mean) * 180 / np.pi:.6f} deg\n")
            f.write(f"ΔB_p  = {abs(bp1_mean - bp2_mean):.6f}\n\n")

        f.write("===== NORMALIZATION CHECK =====\n")
        f.write(f"Sum posterior = {np.sum(posterior):.6e}\n")
    # ==============================================================
    # PLOT
    # ==============================================================

    P_beta_i = np.sum(posterior, axis=2)
    P_beta_bp = np.sum(posterior, axis=1)
    P_i_bp = np.sum(posterior, axis=0)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    params = [beta_vector, i_vector, bp]
    labels = [r"$\beta$", r"$i$", r"$B_p$"]
    P_1D = [P_beta, P_i, P_bp]

    for i in range(3):
        axes[i, i].plot(params[i], P_1D[i])
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

    plt.tight_layout()

    out_pdf = os.path.join("results", base + ".pdf")
    plt.savefig(out_pdf)
    plt.close()

    plt.plot(bp, P_bp)
    plt.xlabel(r"$B_{p}$", fontsize=20)
    plt.ylabel(r"$P$", fontsize=20)
    plt.tight_layout()
    plt.title(r'Posterior $B_{p}$', fontsize=20)
    out_pdf = os.path.join("results", base + "_bp.pdf")
    plt.savefig(out_pdf)
    plt.close()

    plt.plot(beta_vector, P_beta)
    plt.xlabel(r"$\beta$", fontsize=20)
    plt.ylabel(r"$P$", fontsize=20)
    plt.tight_layout()
    plt.title(r'Posterior $\beta$', fontsize=20)
    out_pdf = os.path.join("results", base + "_beta.pdf")
    plt.savefig(out_pdf)
    plt.close()

    plt.plot(i_vector, P_i)
    plt.xlabel(r"$i$", fontsize=20)
    plt.ylabel(r"$P$", fontsize=20)
    plt.tight_layout()
    plt.title(r'Posterior $i$', fontsize=20)
    out_pdf = os.path.join("results", base + "_i.pdf")
    plt.savefig(out_pdf)
    plt.close()

    return base  # можно вернуть что-то для статистики

if __name__ == "__main__":

    files = glob.glob("bfield_data/*.csv")

    print(f"Found {len(files)} files")

    t0 = time.time()

    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(process_one_file, files)

    t1 = time.time()

    print(f"Total time: {t1 - t0:.2f} s")