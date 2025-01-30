import coffea.util
import sys
import mplhep as hep
import os

import matplotlib.pyplot as plt

hep.style.use(hep.style.CMS)

path_dict = {
        "pT variance" : "/work/tharte/datasets/sample_spanet/loose_selection_random_pt/output_all.coffea",
        "pT and mass variance" : "/work/tharte/datasets/sample_spanet/loose_selection_random_pt_mass/output_all.coffea",
        "pT and mass variance wide" : "/work/tharte/datasets/sample_spanet/loose_selection_random_pt_mass_wide/output_all.coffea",
        }
for key, value in path_dict.items():
    path_dict[key] = coffea.util.load(value)["variables"]

#paths = sys.argv[1:]
#cfiles = [coffea.util.load(path) for path in paths]
#output = "."

data_categories = [
    "4b_region",
#    "4b_delta_Dhh_above_30",
    "2b_region",
#    "2b_delta_Dhh_above_30",
]


#subfiles = [cfile["variables"] for cfile in cfiles]

higgs1 = {}
higgs2 = {}
pt_matched = {}
for type_key, value in path_dict.items():
    print(type_key)
    higgs1_temp = []
    higgs2_temp = []
    pt_matched_temp = []
    for key, data in value.items():
        key1 = list(data.keys())[0]
        key2 = list(data[key1].keys())[0]
        if "Higgs2" in key:
            higgs2_temp.append(data[key1][key2])
        elif "Higgs1" in key:
            higgs1_temp.append(data[key1][key2])
        elif "pt_" in key:
            pt_matched_temp.append(data[key1][key2])
    higgs1[type_key] = higgs1_temp
    higgs2[type_key] = higgs2_temp
    pt_matched[type_key] = pt_matched_temp

print(higgs1)
print(higgs2)

for cat in data_categories:
    for higgs, plotname in zip([higgs1, higgs2, pt_matched], ["Higgs1", "Higgs2", "pT jets"]):
        fig, ax = plt.subplots(figsize=(10, 8))
        print_cat = cat.split('_')
        print_cat = ' '.join(print_cat)
        fig.suptitle(f"{print_cat}")

        hep.cms.label(loc=0, data=True, label="Preliminary", lumi=26.67, com=13.6)
        print(higgs)
        for type_key, higgs_type in higgs.items():
            print(type_key)
            for higgs_axis in higgs_type:
                values = higgs_axis[{"cat": cat}][{"variation": "nominal"}].values()
                values = [value/sum(values) for value in values]
                edges = higgs_axis[{"cat": cat}][{"variation": "nominal"}].axes[0].edges
                name = higgs_axis[{'cat': cat}][{'variation': 'nominal'}].axes[0].name
                print(name)
                hep.histplot(
                    (values, edges),
                    stack=False,
                    histtype="step",
                    alpha=0.5,
                    label=f"{name[:-5]} {type_key}",
                )

            # ax.step(edges[0:-1], values, where="post", label=f"{name[:-5]}")
        plt.tight_layout()
        if plotname == "pT jets":
            ax.set_xlabel("$pT$ [GeV]")
        else:
            ax.set_xlabel("$M_{H}$ [GeV]")
        ax.set_ylabel("Counts normalised")
        # ax.set_ylabel("Counts")
        ax.legend()
        os.makedirs(f"plots/{cat}/{cat}_{plotname}", exist_ok=True)
        plt.savefig(f"plots/{cat}/{cat}_{plotname}")
