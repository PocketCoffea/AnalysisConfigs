from coffea.util import load
from pocket_coffea.parameters.histograms import *


import os
localdir = os.path.dirname(os.path.abspath(__file__))

o = load("out_test/output_all.coffea")
print(o.keys())
print(o["cutflow"])

# # Loading default parameters
# from pocket_coffea.parameters import defaults
# default_parameters = defaults.get_default_parameters()
# defaults.register_configuration_dir("config_dir", localdir+"/params")

# # Samples to exclude in specific histograms
# exclude_data = ["DATA_SingleEle", "DATA_SingleMuon"]
# exclude_nonttbar = ["ttHTobb", "TTTo2L2Nu", "SingleTop", "WJetsToLNu_HT"] + exclude_data

# # adding object preselection
# year = "2018"
# parameters = defaults.merge_parameters_from_files(default_parameters,
#                                                   f"{localdir}/params/object_preselection.yaml",
#                                                   #f"{localdir}/params/plotting_style.yaml",
#                                                   update=True)


# from pocket_coffea.utils.plot_utils import PlotManager, Shape
# plotter = PlotManager(
#     variables=o["variables"].keys(),
#     hist_objs=o["variables"],
#     datasets_metadata=o['datasets_metadata'],
#     plot_dir="plots",
#     style_cfg=parameters['plotting_style'],
#     only_cat=None,
#     workers=8,
#     log=False,
#     density=True,
#     save=True,
#     toplabel="4.41 $fb^{-1}$, $\sqrt{s}=$8 TeV, 2012 C"
# )
# print(plotter.shape_objects)

# shape = plotter.shape_objects["JetMatched_pt"]

# fig, ax = shape.define_figure(ratio=False)
# shape.plot_datamc("baseline", ratio=False)

# ax.set_xlim(50, 150)
# fig.savefig("test.png")

# #plot a figure
# plotter.plot("JetMatched_pt", "baseline", ratio=False, density=True)


response=o["columns"]["QCD"]["QCD_PT-15to7000_PT-15to7000_2018"]["baseline"]["JetMatched_Response"]
print(response)
# reponse is an column_accumulator with an array inside. Create an histogram in matplotlib from it
import matplotlib.pyplot as plt
import awkward as ak

a=ak.flatten(response)
print(a)


# plt.hist(res_a, bins=100)
# #save
# plt.savefig("test.png")