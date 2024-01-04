from coffea.util import load
from pocket_coffea.parameters.histograms import *
from pocket_coffea.utils.plot_efficiency import *
import numpy as np
import matplotlib.pyplot as plt
import os
from params.binning import *

import argparse

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

response_bins = np.linspace(0, 4, 1000)
response_bins_mid = (response_bins[1:] + response_bins[:-1]) / 2


# medians = np.zeros((len(eta_bins) - 1, len(pt_bins) - 1))
medians= list(list())

main_dir = "out_cartesian_histo2d_complete"
o = load(f"{main_dir}/output_all.coffea")
var = f"MatchedJets_ResponseVSpt"
histos_dict = o["variables"][var]

for sample in histos_dict.keys():
    for dataset in histos_dict[sample].keys():
        # histo = np.array(histos_dict[sample][dataset][cat][0, 0, :])["value"]
        histo = histos_dict[sample][dataset]
        # print(histo)
        categories=list(histo.axes["cat"])
        variations=list(histo.axes["variation"])
        for i in range(len(categories) - 1):
            medians.append(list())
            for var in  variations:
                h=histo[{'cat': categories[i]}][{'variation': var}]
                # h is a histo2d and we want to find the median of the distribution along the axis MatchedJets.Response
                # for each bin in the axis MatchedJets.pt
                # so we need to loop over the bins in the axis MatchedJets.pt
                for j in range(len(h.axes["MatchedJets.pt"])):
                    # get the histo1d for the bin j in the axis MatchedJets.pt
                    h1d=h[{'MatchedJets.pt': j}]
                    # get the values of the histo1d
                    values=h1d.values()
                    # print("values", values)
                    # get the bins of the histo1d
                    bins=h1d.axes[0].edges
                    # print("bins", bins)
                    # find the bin which is the median of the histogram
                    cdf = np.cumsum(values)
                    # print("cdf", cdf)
                    cdf_normalized = cdf / cdf[-1]
                    median_bin_index = np.argmax(cdf_normalized >= 0.5)
                    median= bins[median_bin_index]
                    medians[i].append(median)
                    # medians[i,j] = median



            # find the bin which is the median of the histogram
            # cdf = np.cumsum(histo)
            # cdf_normalized = cdf / cdf[-1]
            # median_bin_index = np.argmax(cdf_normalized >= 0.5)
            # median= response_bins_mid[median_bin_index]
        # medians[i,j] = median

# for i in range(len(eta_bins) - 1):
#     cat = f"MatchedJets_eta{eta_bins[i]}to{eta_bins[i+1]}"
#     for j in range(len(pt_bins) - 1):
#         for sample in histos_dict.keys():
#             for dataset in histos_dict[sample].keys():
#                 histo = np.array(histos_dict[sample][dataset][cat][0, 0, :])["value"]
#                 print(histo)
#                 # find the bin which is the median of the histogram
#                 cdf = np.cumsum(histo)
#                 cdf_normalized = cdf / cdf[-1]
#                 median_bin_index = np.argmax(cdf_normalized >= 0.5)  # find where CDF reaches 0.5
#                 median= response_bins_mid[median_bin_index]
#                 medians[i,j] = median

print("binned", medians)
median_dir = f"{main_dir}/median_plots_binned"
os.makedirs(f"{median_dir}", exist_ok=True)


# do the same for unbinned
# fig, ax = plt.subplots()
# im = ax.imshow(medians_un, cmap="viridis")
# ax.set_xticks(np.arange(len(pt_bins)-1))
# ax.set_yticks(np.arange(len(eta_bins)-1))
# ax.set_xticklabels(pt_bins[1:])
# ax.set_yticklabels(eta_bins[1:])
# ax.set_xlabel("pt")
# ax.set_ylabel("eta")
# plt.colorbar(im)
# # plt.show()
# fig.savefig(f"{median_dir}/median_2d_unbinned.png")


# do the same for unbinned
fig, ax = plt.subplots()
for i in range(len(eta_bins) - 1):
    ax.plot(pt_bins[1:], medians[i][:], label=f"eta{eta_bins[i]}to{eta_bins[i+1]}")
ax.set_xlabel("pt")
ax.set_ylabel("median")
# log x scale
ax.set_xscale("log")
ax.legend()
# plt.show()
fig.savefig(f"{median_dir}/median_1d_binned.png")

# create a plot for each eta bin with the median as a function of pt
for i in range(len(eta_bins) - 1):
    fig, ax = plt.subplots()
    ax.plot(pt_bins[1:], medians[i][:], label=f"eta{eta_bins[i]}to{eta_bins[i+1]}")
    ax.set_xlabel("pt")
    ax.set_ylabel("median")
    # log x scale
    ax.set_xscale("log")
    ax.legend()
    # plt.show()
    fig.savefig(f"{median_dir}/median_1d_eta{eta_bins[i]}to{eta_bins[i+1]}.png")
    fig.clf()

print(median_dir)