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
from workflow import VBFHH4bbQuarkMatchingProcessor

import custom_cut_functions
import custom_cuts
from custom_cut_functions import *
from custom_cuts import *


import os

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = "2022_postEE"
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    # f"{localdir}/params/plotting_style.yaml",
    update=True,
)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/QCD.json",
            f"{localdir}/datasets/signal_VBF_HH4b_local.json",
        ],
        "filter": {
            "samples": (
                [
                    "VBF_HHto4B",
                    #TODO qcd
                ]

            ),
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=VBFHH4bbQuarkMatchingProcessor,
    workflow_options={
        "parton_jet_min_dR": 0.4,
        "max_num_jets": 5,
        "which_bquark": "last",
    },
    skim=[
        get_HLTsel(primaryDatasets=["JetMET"]),
    ],
    preselections=[
        hh4b_presel
    ],
    categories={
        "4b_region": [hh4b_4b_region],  
        "4b_region_05qvg": [hh4b_4b_region, qvg_05_region],
        "4b_region_06qvg": [hh4b_4b_region, qvg_06_region],
        "4b_region_07qvg": [hh4b_4b_region, qvg_07_region],
        "4b_region_08qvg": [hh4b_4b_region, qvg_08_region],
        "4b_region_09qvg": [hh4b_4b_region, qvg_09_region],
        # "2b_region": [hh4b_2b_region],
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
        **jet_hists(coll="JetGood", pos=0),
        **jet_hists(coll="JetGood", pos=1),
        **jet_hists(coll="JetGood", pos=2),
        **jet_hists(coll="JetGood", pos=3),
        **jet_hists(coll="JetGood", pos=4),
        **jet_hists(coll="JetGood", pos=5),
        **{
            f"JetGoodVBFeta": HistConf(
                [
                    Axis(
                        coll=f"JetGoodVBF",
                        field="eta",
                        bins=60,
                        start=-5,
                        stop=5,
                        label=f"JetGoodVBFeta",
                    )
                ]
            )
        },
        **{
            f"JetGoodVBFQvG": HistConf(
                [
                    Axis(
                        coll=f"JetGoodVBF",
                        field="btagPNetQvG",
                        pos = 0,
                        bins=60,
                        start=0,
                        stop=1,
                        label=f"JetGoodVBFQvG",
                    )
                ]
            )
        },
        **{
            f"JetGoodVBFQvG": HistConf(
                [
                    Axis(
                        coll=f"JetGoodVBF",
                        field="btagPNetQvG",
                        pos = 1,
                        bins=60,
                        start=0,
                        stop=1,
                        label=f"JetGoodVBFQvG",
                    )
                ]
            )
        },
        **{
            f"JetGoodVBFdeltaEta": HistConf(
                [
                    Axis(
                        coll=f"events",
                        field="deltaEta",
                        bins=60,
                        start=5,
                        stop=10,
                        label=f"JetGoodVBFQvG",
                    )
                ]
            )
        },
    },
    columns={
        "common": {
            "inclusive": (
                [
                    ColOut(
                        "JetGood",
                        [
                            "pt",
                            "eta",
                            "phi",
                            "mass",
                            "btagPNetB",
                            "hadronFlavour",
                        ],
                    ),
                ]

            ),
        },
        "bysample": {},
    },
)
