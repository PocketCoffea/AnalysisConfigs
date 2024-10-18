from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.cut_functions import get_nObj_eq, get_nObj_min, get_HLTsel, get_nBtagMin, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.workflows.tthbb_base_processor import ttHbbBaseProcessor

import custom_cut_functions
import custom_cuts
from custom_cut_functions import *
from custom_cuts import *

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection_semileptonic.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  update=True)

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [
                  f"{localdir}/datasets/backgrounds_MC.json",
                  ],
        "filter" : {
            "samples": [
                        "SingleTop",
                        "WJetsToLNu_HT",
                        "DYJetsToLL",
                        "VV",
                        "TTV",
                        ],
            "samples_exclude" : [],
            "year": ["2016_PreVFP",
                     "2016_PostVFP",
                     "2017",
                     "2018"
                     ] #All the years
        },
        "subsamples": {}
    },

    workflow = ttHbbBaseProcessor,
    workflow_options = {},
    
    save_skimmed_files = "root://t3dcachedb03.psi.ch:1094/pnfs/psi.ch/cms/trivcat/store/user/mmarcheg/ttHbb/skim/4j2b/",
    skim = [get_nPVgood(1),
            eventFlags,
            goldenJson,
            get_nObj_min(4, 15., "Jet"),
            get_nBtagMin(2, 15., coll="Jet", wp="M"),
            get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon"])],
    
    preselections = [passthrough],
    categories = {
        "baseline": [passthrough],
    },
    weights= {
        "common": {
            "inclusive": [
            ],
            "bycategory": {}
        },
    },
    variations = {
        "weights": {
            "common": {
                "inclusive": [
                              ],
                "bycategory": {}
            },
        },
    },
    
    variables = {
    },
)

# Registering custom functions
import cloudpickle
cloudpickle.register_pickle_by_value(custom_cut_functions)
cloudpickle.register_pickle_by_value(custom_cuts)
