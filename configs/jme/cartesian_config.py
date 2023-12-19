from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import (
    get_nObj_eq,
    get_nObj_min,
    get_nObj_less,
    get_HLTsel,
    get_nBtagMin,
    get_nElectron,
    get_nMuon,
)
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.categorization import CartesianSelection, MultiCut

import workflow
from workflow import QCDBaseProcessor

import custom_cut_functions
from cuts import *

# import custom_cuts
from custom_cut_functions import *
from params.binning import *

# from custom_cuts import *
# from params.binning import bins
import os

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# Samples to exclude in specific histograms
# exclude_data = ["DATA_SingleEle", "DATA_SingleMuon"]
# exclude_nonttbar = ["ttHTobb", "TTTo2L2Nu", "SingleTop", "WJetsToLNu_HT"] + exclude_data

# adding object preselection
year = "2018"
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    # f"{localdir}/params/triggers.yaml",
    # f"{localdir}/params/lepton_scale_factors.yaml",
    # f"{localdir}/params/plotting_style.yaml",
    update=True,
)


# cuts_pt = []
# cuts_names_pt = []
# for j in range(len(pt_bins) - 1):
#     pt_low, pt_high = pt_bins[j], pt_bins[j + 1]
#     cuts_pt.append(get_ptbin(pt_low, pt_high))
#     cuts_names_pt.append(f'pt{pt_low}to{pt_high}')

cuts_eta = []
cuts_names_eta = []
for i in range(len(eta_bins) - 1):
    eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
    cuts_eta.append(get_etabin(eta_low, eta_high))
    cuts_names_eta.append(f"GenJet_eta{eta_low}to{eta_high}")


multicuts = [
    MultiCut(name="eta", cuts=cuts_eta, cuts_names=cuts_names_eta),
    # MultiCut(name="pt",
    #          cuts=cuts_pt,
    #          cuts_names=cuts_names_pt),
]

common_cats = {
    "baseline": [passthrough],
}

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/QCD.json",
            # f"{localdir}/datasets/DATA_SingleMuon.json",
        ],
        "filter": {
            "samples": [
                "QCD",
                # "DATA_SingleMuon",
            ],
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=QCDBaseProcessor,
    # workflow_options={"parton_jet_min_dR": 0.3},
    skim=[
        # get_nObj_min(4, 15.0, "Jet"),
        # get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon"]),
    ],
    preselections=[],
    categories=CartesianSelection(multicuts=multicuts, common_cats=common_cats),
    # categories={
    #             **common_cats,
    #     # **eta_cuts,
    # },
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {},
            },
            "bysample": {},
        }
    },
    variables={
        **{
            "GenJet": HistConf(
                [
                    Axis(
                        coll="Jet",
                        field="pt",
                        bins=100,
                        start=0,
                        stop=100,
                        label="GenJet_pt",
                    )
                ]
            ),
            # "GenJet_Response": HistConf(
            #     [
            #         Axis(
            #             coll="GenJet",
            #             field="Response",
            #             bins=100,
            #             start=0,
            #             stop=4,
            #             pos=None,
            #             label="GenJet_Response",
            #         )
            #     ]
            # ),
            # "GenJet_DeltaR": HistConf(
            #     [
            #         Axis(
            #             coll="GenJet",
            #             field="DeltaR",
            #             bins=100,
            #             start=0,
            #             stop=0.5,
            #             pos=None,
            #             label="GenJet_DeltaR",
            #         )
            #     ]
            # ),
            "GenJet_eta": HistConf(
                [
                    Axis(
                        coll="Jet",
                        field="eta",
                        bins=100,
                        start=-5.5,
                        stop=5.5,
                        pos=None,
                        label="GenJet_eta",
                    )
                ]
            ),
        },
        # # plot variables in eta bins and pt bins
        # **{
        #     f"GenJet_pt{pt_bins[j]}to{pt_bins[j+1]}_{var}": HistConf(
        #         [
        #             Axis(
        #                 coll=f"GenJet_pt{pt_bins[j]}to{pt_bins[j+1]}",
        #                 field=var,
        #                 bins=100,
        #                 start=start,
        #                 stop=stop,
        #                 label=f"GenJet_pt{pt_bins[j]}to{pt_bins[j+1]}_{var}",
        #             )
        #         ]
        #     )
        #     for j in range(len(pt_bins) - 1)  # for each pt bin
        #     for var, start, stop in zip(
        #         # ["Response"],
        #         # [4]
        #         ["pt", "eta", "Response"],
        #         [0, -5.5, 0],
        #         [100, 5.5, 4],
        #     )
        # },
    },
    columns={
        "common": {
            # "inclusive": [
            #     ColOut("GenJet", ["Response", "pt", "eta"]),
            # ]
            # + [
            #     ColOut(
            #         f"GenJet_pt{pt_bins[j]}to{pt_bins[j+1]}",
            #         ["Response", "pt", "eta"],
            #         # fill_none=False,
            #     )
            #     for j in range(len(pt_bins) - 1)  # for each pt bin
            # ],

            # cat: [ColOut("GenJet", ["Response", "pt", "eta"])]
            # + [
            #     ColOut(
            #         f"GenJet_pt{pt_bins[j]}to{pt_bins[j+1]}",
            #         ["Response", "pt", "eta"],
            #         # fill_none=False,
            #     )
            #     for j in range(len(pt_bins) - 1)  # for each pt bin
            # ]
            # for cat in cuts_names_eta
        },
        "bysample": {
            "QCD": {
                "bycategory": {
                    "baseline": [
                        ColOut("Jet", ["pt", "eta"], fill_none=False),
                        # ColOut("GenJet", ["Response", "pt", "eta"]),
                    ]
                    # + [
                    #     ColOut(
                    #         f"GenJet_pt{pt_bins[j]}to{pt_bins[j+1]}",
                    #         ["Response", "pt", "eta"],
                    #         # fill_none=False,
                    #     )
                    #     for j in range(len(pt_bins) - 1)  # for each pt bin
                    # ],

                    # cat: [ColOut("GenJet", ["pt", "eta"])]
                    # + [
                    #     ColOut(
                    #         f"GenJet_pt{pt_bins[j]}to{pt_bins[j+1]}",
                    #         ["Response", "pt", "eta"],
                    #     )
                    #     for j in range(len(pt_bins) - 1)  # for each pt bin
                    # ]
                    # for cat in cuts_names_eta
                }
            },
        },
    },
)

run_options = {
    "executor": "dask/slurm",
    "env": "conda",
    "workers": 1,
    "scaleout": 50,
    "worker_image": "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-cc7-latest",
    "queue": "standard",
    "walltime": "12:00:00",  # 00:40:00
    "mem_per_worker": "6GB",  # 4GB
    "disk_per_worker": "1GB",
    "exclusive": False,
    "chunk": 400000,
    "retries": 50,
    "treereduction": 5,  # 20,
    "adapt": False,
}


if "dask" in run_options["executor"]:
    import cloudpickle

    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(custom_cut_functions)
    # cloudpickle.register_pickle_by_value(custom_cuts)
