import coffea.util
import sys
import mplhep as hep
import os

import matplotlib.pyplot as plt

hep.style.use(hep.style.CMS)

path = sys.argv[1]
trained = sys.argv[2]
cfile = coffea.util.load(path)

data_categories = [
    "4b_region",
    "4b_delta_Dhh_above_30",
    "2b_region",
    "2b_delta_Dhh_above_30",
]


subfile = cfile["variables"]

print(subfile.keys())
higgs1 = []
higgs2 = []
pt_matched = []
for key, data in subfile.items():
    key1 = list(data.keys())[0]
    key2 = list(data[key1].keys())[0]
    if "Higgs2" in key:
        higgs2.append(data[key1][key2])
    elif "Higgs1" in key:
        higgs1.append(data[key1][key2])
    elif "pt_" in key:
        print(key)
        print(key1)
        print(key2)
        pt_matched.append(data[key1][key2])

print(cfile.keys())

print(higgs1)
print(higgs2)

for cat in data_categories:
    for higgs, plotname in zip([higgs1, higgs2, pt_matched[0:]], ["Higgs1", "Higgs2", "pT jets"]):
        fig, ax = plt.subplots(figsize=(10, 8))
        print_cat = cat.split('_')
        print_cat = ' '.join(print_cat)
        fig.suptitle(f"{print_cat}")

        hep.cms.label(loc=0, data=True, label="Preliminary", lumi=26.67, com=13.6)
        for higgs_axis in higgs:
            if trained:
                values = higgs_axis[{"cat": cat}].values()
                # values = [value/sum(values) for value in values]
                edges = higgs_axis[{"cat": cat}].axes[0].edges
                name = higgs_axis[{"cat": cat}].axes[0].name
            else:
                values = higgs_axis[{"cat": cat}][{"variation": "nominal"}].values()
                ## values = [value/sum(values) for value in values]
                edges = higgs_axis[{"cat": cat}][{"variation": "nominal"}].axes[0].edges
                name = higgs_axis[{"cat": cat}][{"variation": "nominal"}].axes[0].name
            print(name)
            print(values)
            print(edges)
            hep.histplot(
                (values, edges),
                stack=False,
                histtype="step",
                alpha=0.5,
                label=f"{name[:-5]}",
            )

            # ax.step(edges[0:-1], values, where="post", label=f"{name[:-5]}")
        plt.tight_layout()
        if plotname == "pT jets":
            ax.set_xlabel("$pT$ [GeV]")
        else:
            ax.set_xlabel("$M_{H}$ [GeV]")
        # ax.set_ylabel("Counts normalised")
        ax.set_ylabel("Counts")
        ax.legend()
        os.makedirs(f"plots/{cat}/{cat}_{plotname}", exist_ok=True)
        plt.savefig(f"plots/{cat}/{cat}_{plotname}")
