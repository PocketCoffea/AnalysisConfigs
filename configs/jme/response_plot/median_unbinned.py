from coffea.util import load
from pocket_coffea.parameters.histograms import *
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
import mplhep as hep

sys.path.append("../")
from params.binning import *

parser = argparse.ArgumentParser(description="Run the jme analysis")
parser.add_argument(
    "-c",
    "--cartesian",
    action="store_true",
    help="Run cartesian multicuts",
    default=False,
)
parser.add_argument(
    "-l",
    "--load",
    action="store_true",
    help="Load medians from file",
    default=False,
)
args = parser.parse_args()

localdir = os.path.dirname(os.path.abspath(__file__))


main_dir = "../out_separate_eta_bin_seq" if not args.cartesian else "out_cartesian"
median_dir = f"{main_dir}/median_plots_unbinned"

if args.load:
    medians = np.load(f"{median_dir}/medians.npy")
    err_medians = np.load(f"{median_dir}/err_medians.npy")
    print("loaded", medians, err_medians)

else:
    medians = np.zeros((len(eta_bins) - 1, len(pt_bins) - 1))
    err_medians = np.zeros((len(eta_bins) - 1, len(pt_bins) - 1))
    o_cartesian = load(f"{main_dir}/output_all.coffea") if args.cartesian else None
    num_tot = 0
    for i in range(len(eta_bins) - 1):
        file_name = f"{main_dir}/eta{eta_bins[i]}to{eta_bins[i+1]}/output_all.coffea"
        o = load(file_name) if not args.cartesian else o_cartesian
        cat = (
            "baseline"
            if not args.cartesian
            else f"MatchedJets_eta{eta_bins[i]}to{eta_bins[i+1]}"
        )
        # histos_dict = o["variables"]
        columns_dict = o["columns"]
        # median_dict[f"eta{eta_bins[i]}to{eta_bins[i+1]}"] = {}
        for j in range(len(pt_bins) - 1):
            # var_histo = f"MatchedJets_Response_eta{eta_bins[i]}to{eta_bins[i+1]}_pt{pt_bins[j]}to{pt_bins[j+1]}"
            var = (
                f"MatchedJets_eta{eta_bins[i]}to{eta_bins[i+1]}_pt{pt_bins[j]}to{pt_bins[j+1]}_Response"
                if not args.cartesian
                else f"MatchedJets_pt{pt_bins[j]}to{pt_bins[j+1]}_Response"
            )
            # print(f"variable {var} ")
            for sample in columns_dict.keys():
                for dataset in columns_dict[sample].keys():
                    # print(f"variable {var} sample {sample} dataset {dataset}")
                    # histo = np.array(histos_dict[var_histo][sample][dataset][0, 0, :])["value"]
                    column = columns_dict[sample][dataset][cat][var].value
                    # sort column in ascending order
                    column = np.sort(column[column != -999.0])
                    median = np.median(column)
                    num_tot += len(column)
                    medians[i, j] = median
                    # print("median:\n", median)

                    # compute the error on the median as 1.253 * hist->GetRMS() / TMath::Sqrt(hist->GetEffectiveEntries()); but for an unbinned distribution
                    mean = np.mean(column)
                    rms = np.sqrt(np.mean((column - mean) ** 2))
                    err_median = 1.253 * rms / np.sqrt(len(column))
                    err_medians[i, j] = err_median
                    # print("err_median:\n", err_median)

    print("medians", medians)
    print("err_medians", err_medians)
    os.makedirs(f"{median_dir}", exist_ok=True)
    print("num_tot", num_tot)

    # save medians to file
    np.save(f"{median_dir}/medians.npy", medians)
    np.save(f"{median_dir}/err_medians.npy", err_medians)



# do the same for unbinned
# fig, ax = plt.subplots()
# for i in range(len(eta_bins) - 1):
#     # ax.plot(pt_bins[1:], medians[i, :], label=f"eta{eta_bins[i]}to{eta_bins[i+1]}", marker="o")
#     ax.errorbar(
#         pt_bins[1:],
#         medians[i, :],
#         yerr=err_medians[i, :],
#         label=f"eta{eta_bins[i]}to{eta_bins[i+1]}",
#         marker="o",
#     )
# ax.set_xlabel("pt")
# ax.set_ylabel("median")
# # log x scale
# ax.set_xscale("log")
# ax.legend()
# # plt.show()
# fig.savefig(f"{median_dir}/median_1d.png")

# create a plot for each eta bin with the median as a function of pt
for i in range(len(eta_bins) - 1):
    fig, ax = plt.subplots()

    ax.errorbar(
        pt_bins[1:],
        medians[i, :],
        yerr=err_medians[i, :],
        label=f"{eta_bins[i]} <" + r"$\eta$" + f"< {eta_bins[i+1]}",
        marker=".",
        color="black",
        linestyle="None",
    )
    # write axis name in latex
    ax.set_xlabel(r"$p_{T}^{Gen}$ [GeV]")
    ax.set_ylabel("Median (Response) unbinned")
    # log x scale
    ax.set_xscale("log")
    # remove border of legend
    ax.legend(frameon=False)

    plt.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
    # hep.style.use("CMS")
    hep.cms.label(year="2022", com="13.6", label="Private Work")

    # plt.show()
    fig.savefig(
        f"{median_dir}/median_1d_eta{eta_bins[i]}to{eta_bins[i+1]}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)