from coffea.util import load
from pocket_coffea.parameters.histograms import *
from pocket_coffea.utils.plot_efficiency import *
import numpy as np
import matplotlib.pyplot as plt
import os
import mplhep as hep
import argparse
import sys

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
args = parser.parse_args()

localdir = os.path.dirname(os.path.abspath(__file__))

# o = load("out_mc/output_all.coffea")
# columns_dict = o["columns"]
# response_bins = np.array(response_bins)
# response_bins_mid = (response_bins[1:] + response_bins[:-1]) / 2


# medians = np.zeros((len(eta_bins) - 1, len(pt_bins) - 1))
medians = list(list())
err_medians = list(list())

main_dir = "../out_cartesian_neg"
o = load(f"{main_dir}/output_all.coffea")
var = f"MatchedJets_ResponseVSpt"
histos_dict = o["variables"][var]

for sample in histos_dict.keys():
    for dataset in histos_dict[sample].keys():
        # histo = np.array(histos_dict[sample][dataset][cat][0, 0, :])["value"]
        histo = histos_dict[sample][dataset]
        # print(histo)
        categories = list(histo.axes["cat"])
        print("categories", categories)

        # remove the baseline category
        categories.remove("baseline") if "baseline" in categories else None

        # order the categories so that the ranges in eta are increasing
        categories = sorted(
            categories, key=lambda x: float(x.split("eta")[1].split("to")[0])
        )
        variations = list(histo.axes["variation"])
        for i in range(len(categories)):
            medians.append(list())
            err_medians.append(list())
            for var in variations:
                h = histo[{"cat": categories[i]}][{"variation": var}]
                # h is a histo2d and we want to find the median of the distribution along the axis MatchedJets.Response
                # for each bin in the axis MatchedJets.pt
                # so we need to loop over the bins in the axis MatchedJets.pt
                for j in range(len(h.axes["MatchedJets.pt"])):
                    # print("\n\n eta", categories[i], "pt", h.axes["MatchedJets.pt"][j])
                    # get the histo1d for the bin j in the axis MatchedJets.pt
                    h1d = h[{"MatchedJets.pt": j}]
                    # get the values of the histo1d
                    values = h1d.values()
                    if all(v == 0 for v in values):
                        # print("all values are 0")
                        medians[i].append(np.nan)
                        err_medians[i].append(np.nan)
                        continue
                    # print("values", values)
                    # get the bins of the histo1d
                    bins = h1d.axes[0].edges
                    # print("bins", bins)
                    # find the bin which is the median of the histogram
                    cdf = np.cumsum(values)
                    # # print("cdf", cdf)
                    cdf_normalized = cdf / cdf[-1]
                    # # print("cdf_normalized", cdf_normalized)
                    median_bin_index = np.argmax(cdf_normalized >= 0.5)
                    # # print("median_bin_index", median_bin_index)
                    bins_mid = (bins[1:] + bins[:-1]) / 2
                    median = bins_mid[median_bin_index]
                    # print("median:\n", median)
                    medians[i].append(median if median != 0.0 else np.nan)

                    # print("bins_mid", bins_mid)
                    mean = np.average(bins_mid, weights=values)
                    # print("mean", mean)
                    rms = np.sqrt(np.average((bins_mid - mean) ** 2, weights=values))
                    # print("rms", rms)
                    err_median = 1.253 * rms / np.sqrt(np.sum(values))
                    # print("sum", np.sum(values))
                    # print("err_median:\n", err_median)
                    err_medians[i].append(err_median if err_median != 0.0 else np.nan)


print("medians:\n", medians)
print("err_medians:\n", err_medians)

median_dir = f"{main_dir}/median_plots_binned"
os.makedirs(f"{median_dir}", exist_ok=True)

# save medians to file npy
medians = np.array(medians)
err_medians = np.array(err_medians)
np.save(f"{median_dir}/medians.npy", medians)
np.save(f"{median_dir}/err_medians.npy", err_medians)


correct_eta_bins = []
for cat in categories:
    if "eta" not in cat:
        continue
    print("cat", cat)
    # for each pair of eta bins, find the corresponding category
    eta_min = float(cat.split("eta")[1].split("to")[0])
    eta_max = float(cat.split("eta")[1].split("to")[1])
    # if abs(eta_min) > abs(eta_max):
    #     eta_min, eta_max = eta_max, eta_min
    if eta_min not in correct_eta_bins:
        correct_eta_bins.append(eta_min)
    if eta_max not in correct_eta_bins:
        correct_eta_bins.append(eta_max)

print("correct_eta_bins", correct_eta_bins)


# do the same for unbinned
# fig, ax = plt.subplots()
# for i in range(len(correct_eta_bins) - 1):
#     # ax.plot(pt_bins[1:], medians[i][:], label=f"eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}")
#     ax.errorbar(
#         pt_bins[1:],
#         medians[i, :],
#         yerr=err_medians[i, :],
#         label=f"eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}",
#         marker="o",
#     )
# ax.set_xlabel("pt")
# ax.set_ylabel("median")
# # log x scale
# ax.set_xscale("log")
# ax.legend()
# # plt.show()
# fig.savefig(f"{median_dir}/median_1d_binned.png")

# create a plot for each eta bin with the median as a function of pt
for i in range(len(correct_eta_bins) - 1):
    fig, ax = plt.subplots()

    # ax.plot(pt_bins[1:], medians[i][:], label=f"eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}", marker="o", mfc='orange', mec='orange')
    # ax.errorbar(pt_bins[1:], medians[i,:], yerr=err_medians[i,:], label=f"eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}", marker=".", mfc='orange', mec='orange', linestyle='None')
    # plot the marker black
    ax.errorbar(
        pt_bins[1:],
        medians[i, :],
        yerr=err_medians[i, :],
        label=f"{correct_eta_bins[i]} <" + r"$\eta$" + f"< {correct_eta_bins[i+1]}",
        marker=".",
        color="black",
        linestyle="None",
    )
    # write axis name in latex
    ax.set_xlabel(r"$p_{T}^{Gen}$ [GeV]")
    ax.set_ylabel("Median (Response) binned")
    # log x scale
    ax.set_xscale("log")
    # remove border of legend
    ax.legend(frameon=False)

    plt.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
    # hep.style.use("CMS")
    hep.cms.label(year="2022", com="13.6", label="Private Work")

    # plt.show()
    fig.savefig(
        f"{median_dir}/median_1d_eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)

print(median_dir)
