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
from workflow import HH4bbQuarkMatchingProcessor

import custom_cut_functions
import custom_cuts
from custom_cut_functions import *
from custom_cuts import *

# from params.binning import bins
import os

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = "2018"  # TODO: change year to 2022
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    # f"{localdir}/params/plotting_style.yaml",
    update=True,
)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/DATA_JetMET.json",
            f"{localdir}/datasets/signal_ggF_HH4b_redirector.json",
        ],
        "filter": {
            "samples": [
                # "GluGlutoHHto4B",
                # "GluGlutoHHto4B_kl0_poisson",
                # "GluGlutoHHto4B_kl2p45_poisson",
                # "GluGlutoHHto4B_kl5_poisson",
                # "GluGlutoHHto4B_poisson",
                "DATA_JetMET",
            ],
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=HH4bbQuarkMatchingProcessor,
    workflow_options={
        "parton_jet_min_dR": 0.4,
        "max_num_jets": 4,
        "which_bquark": "last",  # HERE
    },
    skim=[
        get_HLTsel(primaryDatasets=["JetMET"]),
    ],
    preselections=[
        hh4b_presel
    ],
    categories={
        "2b_region": [hh4b_2b_region],
        "4b_region": [hh4b_4b_region],

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
        **count_hist(coll="JetGood", bins=10, start=0, stop=10),
        **count_hist(coll="JetGoodHiggs", bins=10, start=0, stop=10),
        **jet_hists(coll="JetGood", pos=0),
    },
    columns={
        "common": {
            "inclusive": [
                ColOut(
                    "JetGoodHiggs",
                    [
                        "pt",
                        "eta",
                        "phi",
                        "mass",
                        "btagPNetB",
                        "ptPnetRegNeutrino",
                    ],
                ),
                ColOut(
                    "JetGood",
                    [
                        "pt",
                        "eta",
                        "phi",
                        "mass",
                        "btagPNetB",
                        "ptPnetRegNeutrino",
                    ],
                ),
            ],
        },
        "bysample": {},
    },
)

run_options = {
    "executor": "dask/slurm",
    "env": "conda",
    "workers": 1,
    "scaleout": 200,
    "worker_image": "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-cc7-latest",
    "queue": "standard",
    "walltime": "00:20:00",
    "mem_per_worker": "6GB",  # GB
    "disk_per_worker": "1GB",  # GB
    "exclusive": False,
    "chunk": 100000,
    "retries": 50,
    "treereduction": 5,
    "adapt": False,
}


if "dask" in run_options["executor"]:
    import cloudpickle

    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(custom_cut_functions)
    cloudpickle.register_pickle_by_value(custom_cuts)
