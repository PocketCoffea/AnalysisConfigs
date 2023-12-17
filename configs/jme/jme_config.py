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

import workflow
from workflow import QCDBaseProcessor

import custom_cut_functions

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
    categories={
        **{
            "baseline": [passthrough],
        },
        # **eta_cuts,
    },
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
            # **jet_hists(coll="JetGood", pos=0),
            # **jet_hists(coll="JetMatched", pos=0),
            # "GenJet_pt": HistConf(
            #     [
            #         Axis(
            #             coll="GenJetGood",
            #             field="pt",
            #             bins=100,
            #             start=0,
            #             stop=1000,
            #             label="genjet_pt",
            #             pos=0,
            #         )
            #     ]
            # ),
            # "GenJet_eta": HistConf(
            #     [
            #         Axis(
            #             coll="GenJetGood",
            #             field="eta",
            #             bins=100,
            #             start=-5,
            #             stop=5,
            #             label="genjet_eta",
            #             pos=0,
            #         )
            #     ]
            # ),
            # "JetMatched_pt": HistConf(
            #     [
            #         Axis(
            #             coll="JetMatched",
            #             field="pt",
            #             bins=100,
            #             start=0,
            #             stop=100,
            #             label="jet_matched_pt",
            #         )
            #     ]
            # ),
            "MatchedJets_Response": HistConf(
                [
                    Axis(
                        coll="MatchedJets",
                        field="Response",
                        bins=100,
                        start=0,
                        stop=4,
                        pos=None,
                        label="Response",
                    )
                ]
            ),
            # "JetMatched_Response_old": HistConf(
            #     [
            #         Axis(
            #             coll="JetMatched",
            #             field="Response_old",
            #             bins=100,
            #             start=0,
            #             stop=4,
            #             pos=None,
            #             label="Response_old",
            #         )
            #     ]
            # ),
            "MatchedJets_DeltaR": HistConf(
                [
                    Axis(
                        coll="MatchedJets",
                        field="DeltaR",
                        bins=100,
                        start=0,
                        stop=0.5,
                        pos=None,
                        label="DeltaR",
                    )
                ]
            ),
            # "JetMatched_DeltaR_old": HistConf(
            #     [
            #         Axis(
            #             coll="JetMatched",
            #             field="DeltaR_old",
            #             bins=100,
            #             start=0,
            #             stop=0.5,
            #             pos=None,
            #             label="DeltaR_old",
            #         )
            #     ]
            # ),
        },

        # plot variables in eta bins and pt bins
        **{
            f"MatchedJets_{var}_eta{eta_bins[i]}-{eta_bins[i+1]}_pt{pt_bins[j]}-{pt_bins[j+1]}": HistConf(
                [
                    Axis(
                        coll=f"MatchedJets_eta{eta_bins[i]}-{eta_bins[i+1]}_pt{pt_bins[j]}-{pt_bins[j+1]}",
                        field=var,
                        bins=100,
                        start=0,
                        stop=stop,
                        label=f"MatchedJets_{var}_eta{eta_bins[i]}-{eta_bins[i+1]}_pt{pt_bins[j]}-{pt_bins[j+1]}",
                    )
                ]
            )
            for i in range(len(eta_bins) - 1)  # for each eta bin
            for j in range(len(pt_bins) - 1)  # for each pt bin
            for var, stop in zip(
                # ["Response"],
                # [4]
                ["pt", "AbsEta", "Response"],
                [100, 5.5, 4],
            )
        },

    },
    columns={
        "common": {
            "inclusive": [],
        },
        "bysample": {
            "QCD": {
                "bycategory": {
                    "baseline": [
                        ColOut("JetGood", ["pt"]),
                        ColOut("MatchedJets", ["Response", "pt", "DeltaR", "AbsEta"]),
                    ]
                    + [
                        ColOut(
                            f"MatchedJets_eta{eta_bins[i]}-{eta_bins[i+1]}_pt{pt_bins[j]}-{pt_bins[j+1]}",
                            ["Response", "pt", "AbsEta"],
                        )
                        for i in range(len(eta_bins) - 1)  # for each eta bin
                        for j in range(len(pt_bins) - 1)  # for each pt bin
                    ],
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
    "walltime": "00:40:00",
    "mem_per_worker": "4GB",  # GB
    "disk_per_worker": "1GB",  # GB
    "exclusive": False,
    "chunk": 400000,
    "retries": 50,
    "treereduction": 20,
    "adapt": False,
}


if "dask" in run_options["executor"]:
    import cloudpickle

    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(custom_cut_functions)
    # cloudpickle.register_pickle_by_value(custom_cuts)
