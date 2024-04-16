from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import eft_weights

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
        "jsons": [f"{localdir}/datasets/signal_ttHTobb_T2.json",
                  ],
        "filter" : {
            "samples": ["ttHTobb_SM","ttHTobb_BSM"],
            "samples_exclude" : [],
            #"year": []
        },
        "subsamples": {}
    },

    workflow = BaseProcessorGen,
    
    skim = [],
    
    preselections = [passthrough],
    
    categories = {
        "inclusive": [passthrough],
        "weight10": [passthrough]
    },

    weights= {
        "common": {
            "inclusive": [ "genWeight", "XS"],
            "bycategory": {
                "weight10": [ eft_weights.getSMEFTweight(10)]
            },
        },
        "bysample": {},
    },
    variations = {
        "weights": {"common": {"inclusive": [], "bycategory": {}}, "bysample": {}},
    },
    
    variables = {
        "LHE_HT": HistConf(axes = [
            Axis(field="HT", coll="LHE", name="LHE_HT", bins=50, start=0, stop=5000, label="LHE_HT [GeV]") ]),

        
        **jet_hists(coll="GenJet", fields=["pt", "eta", "phi"]),
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
cloudpickle.register_pickle_by_value(eft_weights)
