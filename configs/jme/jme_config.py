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


# read eta_min for the environment variable ETA_MIN
eta_min = float(os.environ.get("ETA_MIN", -999.))
eta_max = float(os.environ.get("ETA_MAX", -999.))

eta_substr=f"_eta{eta_min}to{eta_max}" if (eta_min != -999. and eta_max != -999.) else ""

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
            "MatchedJets_eta": HistConf(
                [
                    Axis(
                        coll="MatchedJets",
                        field="eta",
                        bins=100,
                        start=-5.2,
                        stop=5.2,
                        pos=None,
                        label="eta",
                    )
                ]
            ),
            "MatchedJets_pt": HistConf(
                [
                    Axis(
                        coll="MatchedJets",
                        field="pt",
                        bins=1000,
                        start=0,
                        stop=5000,
                        pos=None,
                        label="pt",
                    )
                ]
            ),

        },
        # plot variables in eta bins and pt bins
        **{
            f"MatchedJets{eta_substr}_pt{pt_bins[j]}to{pt_bins[j+1]}_{var}": HistConf(
                [
                    Axis(
                        coll=f"MatchedJets{eta_substr}_pt{pt_bins[j]}to{pt_bins[j+1]}",
                        field=var,
                        bins=100,
                        start=start,
                        stop=stop,
                        label=f"MatchedJets{eta_substr}_pt{pt_bins[j]}to{pt_bins[j+1]}_{var}",
                    )
                ]
            )
            # for i in range(len(eta_bins) - 1)  # for each eta bin
            for j in range(len(pt_bins) - 1)  # for each pt bin
            for var, start, stop in zip(
                # ["Response"],
                # [4]
                ["pt", "eta", "Response"],
                [0, -5.5, 0],
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
                        ColOut("MatchedJets", ["Response", "pt", "eta"]),
                    ]
                    + [
                        ColOut(
                            f"MatchedJets{eta_substr}_pt{pt_bins[j]}to{pt_bins[j+1]}",
                            ["Response", "pt", "eta"],
                        )
                        # for i in range(len(eta_bins) - 1)  # for each eta bin
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
    "scaleout": 10, #50
    "worker_image": "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-cc7-latest",
    "queue": "standard",
    "walltime": "02:00:00",  # 00:40:00
    "mem_per_worker": "4GB",  # 4GB
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
