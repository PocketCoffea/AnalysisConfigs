from coffea.util import load
from pocket_coffea.parameters.histograms import *
import numpy as np
import matplotlib.pyplot as plt
import os
from params.binning import *


localdir = os.path.dirname(os.path.abspath(__file__))

o = load("out_mc/output_all.coffea")
# print(o.keys())
histos_dict = o["variables"]
columns_dict = o["columns"]

median_dict = {}
response_bins = np.linspace(0, 4, 100)
response_bins_mid = (response_bins[1:] + response_bins[:-1]) / 2

cat="baseline"

medians=np.zeros((len(eta_bins)-1, len(pt_bins)-1))
medians_un=np.zeros((len(eta_bins)-1, len(pt_bins)-1))

for i in range(len(eta_bins)-1):
    median_dict[f"eta{eta_bins[i]}-{eta_bins[i+1]}"] = {}
    for j in range(len(pt_bins)-1):
        var = f"MatchedJets_Response_eta{eta_bins[i]}-{eta_bins[i+1]}_pt{pt_bins[j]}-{pt_bins[j+1]}"
        var_column = f"MatchedJets_eta{eta_bins[i]}-{eta_bins[i+1]}_pt{pt_bins[j]}-{pt_bins[j+1]}_Response"
        for sample in histos_dict[var].keys():
            for dataset in histos_dict[var][sample].keys():
                # print(f"variable {var} sample {sample} dataset {dataset}")
                histo = np.array(histos_dict[var][sample][dataset][0, 0, :])["value"]
                column=columns_dict[sample][dataset][cat][var_column].value
                # sort column in ascending order
                column = np.sort(column[column != -999.])
                print(histo)
                print(column)
                # find the bin which is the median of the histogram
                cdf = np.cumsum(histo)
                cdf_normalized = cdf / cdf[-1]
                median_bin_index = np.argmax(cdf_normalized >= 0.5)  # find where CDF reaches 0.5
                median= response_bins_mid[median_bin_index]
                medians[i,j] = median
                median_un= np.median(column)
                medians_un[i,j] = median_un
                median_dict[f"eta{eta_bins[i]}-{eta_bins[i+1]}"][f"pt{pt_bins[j]}-{pt_bins[j+1]}"] = median

# print(median_dict)
print("binned", medians)
print("unbinned",medians_un)
os.makedirs("median_plots", exist_ok=True)



# do the same for unbinned
fig, ax = plt.subplots()
im = ax.imshow(medians_un, cmap="viridis")
ax.set_xticks(np.arange(len(pt_bins)-1))
ax.set_yticks(np.arange(len(eta_bins)-1))
ax.set_xticklabels(pt_bins[1:])
ax.set_yticklabels(eta_bins[1:])
ax.set_xlabel("pt")
ax.set_ylabel("eta")
plt.colorbar(im)
# plt.show()
fig.savefig("median_plots/median_2d_unbinned.png")


# do the same for unbinned
fig, ax = plt.subplots()
for i in range(len(eta_bins)-1):
    ax.plot(pt_bins[1:], medians_un[i,:], label=f"eta{eta_bins[i]}-{eta_bins[i+1]}")
ax.set_xlabel("pt")
ax.set_ylabel("median")
ax.legend()
# plt.show()
fig.savefig("median_plots/median_1d_unbinned.png")

# create a plot for each eta bin with the median as a function of pt
for i in range(len(eta_bins)-1):
    fig, ax = plt.subplots()
    ax.plot(pt_bins[1:], medians_un[i,:], label=f"eta{eta_bins[i]}-{eta_bins[i+1]}")
    ax.set_xlabel("pt")
    ax.set_ylabel("median")
    ax.legend()
    # plt.show()
    fig.savefig(f"median_plots/median_1d_eta{eta_bins[i]}-{eta_bins[i+1]}.png")


'''
# creat histogram with median values in matplotlib
fig, ax = plt.subplots()
im = ax.imshow(medians, cmap="viridis")
ax.set_xticks(np.arange(len(pt_bins)-1))
ax.set_yticks(np.arange(len(eta_bins)-1))
ax.set_xticklabels(pt_bins[1:])
ax.set_yticklabels(eta_bins[1:])
ax.set_xlabel("pt")
ax.set_ylabel("eta")
plt.colorbar(im)
# plt.show()
fig.savefig("median_plots/median_2d.png")

# for each eta bin, plot the median as a function of pt
fig, ax = plt.subplots()
for i in range(len(eta_bins)-1):
    ax.plot(pt_bins[1:], medians[i,:], label=f"eta{eta_bins[i]}-{eta_bins[i+1]}")
ax.set_xlabel("pt")
ax.set_ylabel("median")
ax.legend()
# plt.show()
fig.savefig("median_plots/median_1d.png")'''