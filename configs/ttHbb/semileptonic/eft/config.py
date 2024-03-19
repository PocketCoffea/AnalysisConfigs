from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *

import workflow
from workflow import BaseProcessorGen

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection_semileptonic.yaml",
                                                  update=True)

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/datasets/signal_ttHTobb_local.json",
                  ],
        "filter" : {
            "samples": ["ttHTobb",
                        ],
            "samples_exclude" : [],
            "year": [
                     "2018"
                     ]
        },
        "subsamples": {}
    },

    workflow = BaseProcessorGen,
    
    skim = [],
    
    preselections = [passthrough],
    categories = {
        "baseline": [passthrough],
    },

    weights= {
        "common": {
            "inclusive": [],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations = {
        "weights": {"common": {"inclusive": [], "bycategory": {}}, "bysample": {}},
    },
    
    variables = {
        **jet_hists(coll="Jet", pos=0)
    },
    columns = {
        "common": {
            "inclusive": [],
            "bycategory": {}
        },
        "bysample": {},
    },
)

# Registering custom functions
import cloudpickle
cloudpickle.register_pickle_by_value(workflow)
