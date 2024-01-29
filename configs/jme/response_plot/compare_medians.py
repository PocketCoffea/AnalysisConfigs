import numpy as np
import matplotlib.pyplot as plt
import os
import mplhep as hep
import sys

sys.path.append("../")
from params.binning import *

dir_median_diff = "median_diff"

dir_unbin = "../out_separate_eta_bin_seq"
medians_unbin = np.load(f"{dir_unbin}/median_plots_unbinned/medians.npy")
err_medians_unbin = np.load(f"{dir_unbin}/median_plots_unbinned/err_medians.npy")

dir_bin_pos = "../out_cartesian_pos"
medians_bin_pos = np.load(f"{dir_bin_pos}/median_plots_binned/medians.npy")
err_medians_bin_pos = np.load(f"{dir_bin_pos}/median_plots_binned/err_medians.npy")

dir_bin_neg = "../out_cartesian_neg"
medians_bin_neg = np.load(f"{dir_bin_neg}/median_plots_binned/medians.npy")
err_medians_bin_neg = np.load(f"{dir_bin_neg}/median_plots_binned/err_medians.npy")

medians_bin_tot = np.concatenate((medians_bin_neg, medians_bin_pos), axis=0)
err_medians_bin_tot = np.concatenate((err_medians_bin_neg, err_medians_bin_pos), axis=0)

os.makedirs(f"{dir_median_diff}", exist_ok=True)
print("medians_unbin", medians_unbin)
print("medians_bin_tot", medians_bin_tot)
print("err_medians_unbin", err_medians_unbin)
print("err_medians_bin_tot", err_medians_bin_tot)
print("dir_median_diff", dir_median_diff)


# plot the difference between the medians bin and unbinned normalized to the unbinned error
diff = medians_unbin - medians_bin_tot
diff_norm = diff / err_medians_unbin

err_diff = np.sqrt(err_medians_bin_tot**2 + err_medians_unbin**2)
err_diff_norm = err_diff / err_medians_unbin

diff_err = err_medians_unbin - err_medians_bin_tot
diff_err_norm = diff_err / err_medians_unbin

for i in range(len(diff_norm)):
    fig, ax = plt.subplots()
    ax.errorbar(
        pt_bins[1:],
        diff_norm[i],
        # yerr=err_diff_norm[i],
        fmt="o",
        label="unbinned - binned / unbinned error",
    )
    # draw line at 0
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("pt")
    ax.set_ylabel("unbinned - binned / unbinned error")
    ax.set_xscale("log")

    eta_min = eta_bins[i]
    eta_max = eta_bins[i + 1]
    print("eta", eta_min, eta_max)

    fig.savefig(f"{dir_median_diff}/median_diff_eta{eta_min}to{eta_max}.png")
    plt.close(fig)

for i in range(len(diff_err_norm)):
    fig, ax = plt.subplots()
    ax.errorbar(
        pt_bins[1:],
        diff_err_norm[i],
        # yerr=err_diff_norm[i],
        fmt="o",
        label="unbinned err - binned err / unbinned error",
    )
    ax.axhline(0, color="black", linestyle="--")

    ax.set_xlabel("pt")
    ax.set_ylabel("unbinned err - binned err/ unbinned error")
    ax.set_xscale("log")

    eta_min = eta_bins[i]
    eta_max = eta_bins[i + 1]
    print("err: eta", eta_min, eta_max)

    fig.savefig(f"{dir_median_diff}/median_diff_err_eta{eta_min}to{eta_max}.png")
    plt.close(fig)