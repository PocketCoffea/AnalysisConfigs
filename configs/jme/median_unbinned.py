from coffea.util import load
from pocket_coffea.parameters.histograms import *
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


medians=np.zeros((len(eta_bins)-1, len(pt_bins)-1))
medians_un=np.zeros((len(eta_bins)-1, len(pt_bins)-1))

main_dir="out_separate_eta_bin" if not args.cartesian else "out_cartesian"
o_cartesian=load(f"{main_dir}/output_all.coffea") if args.cartesian else None
num_tot=0
for i in range(len(eta_bins)-1):
    o=load(f"{main_dir}/eta{eta_bins[i]}to{eta_bins[i+1]}/output_all.coffea") if not args.cartesian else o_cartesian
    cat="baseline" if not args.cartesian else f"MatchedJets_eta{eta_bins[i]}to{eta_bins[i+1]}"
    # histos_dict = o["variables"]
    columns_dict = o["columns"]
    # median_dict[f"eta{eta_bins[i]}to{eta_bins[i+1]}"] = {}
    for j in range(len(pt_bins)-1):
        # var_histo = f"MatchedJets_Response_eta{eta_bins[i]}to{eta_bins[i+1]}_pt{pt_bins[j]}to{pt_bins[j+1]}"
        var = f"MatchedJets_eta{eta_bins[i]}to{eta_bins[i+1]}_pt{pt_bins[j]}to{pt_bins[j+1]}_Response" if not args.cartesian else f"MatchedJets_pt{pt_bins[j]}to{pt_bins[j+1]}_Response"
        for sample in columns_dict.keys():
            for dataset in columns_dict[sample].keys():
                # print(f"variable {var} sample {sample} dataset {dataset}")
                # histo = np.array(histos_dict[var_histo][sample][dataset][0, 0, :])["value"]
                column=columns_dict[sample][dataset][cat][var].value
                # sort column in ascending order
                column = np.sort(column[column != -999.])
                median_un= np.median(column)
                num_tot+=len(column)
                medians_un[i,j] = median_un

print("unbinned",medians_un)
median_dir=f"{main_dir}/median_plots_unbinned"
os.makedirs(f"{median_dir}", exist_ok=True)
print("num_tot", num_tot)



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
for i in range(len(eta_bins)-1):
    ax.plot(pt_bins[1:], medians_un[i,:], label=f"eta{eta_bins[i]}to{eta_bins[i+1]}")
ax.set_xlabel("pt")
ax.set_ylabel("median")
#log x scale
ax.set_xscale('log')
ax.legend()
# plt.show()
fig.savefig(f"{median_dir}/median_1d.png")

# create a plot for each eta bin with the median as a function of pt
for i in range(len(eta_bins)-1):
    fig, ax = plt.subplots()
    ax.plot(pt_bins[1:], medians_un[i,:], label=f"eta{eta_bins[i]}to{eta_bins[i+1]}")
    ax.set_xlabel("pt")
    ax.set_ylabel("median")
    #log x scale
    ax.set_xscale('log')
    ax.legend()
    # plt.show()
    fig.savefig(f"{median_dir}/median_1d_eta{eta_bins[i]}to{eta_bins[i+1]}.png")
    fig.clf()
