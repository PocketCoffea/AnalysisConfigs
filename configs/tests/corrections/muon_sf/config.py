from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters.histograms import *
import workflow
from workflow import MuonProcessor

# Register custom modules in cloudpickle to propagate them to dask workers
import cloudpickle
cloudpickle.register_pickle_by_value(workflow)

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection.yaml",
                                                  f"{localdir}/params/lepton_scale_factors.yaml",
                                                  update=True)



cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/datasets.json",
                    ],
        "filter" : {
            "samples": ["TT_DL"],
            "samples_exclude" : []
        }
    },

    workflow = MuonProcessor,
    
    skim = [], 
    preselections = [passthrough],
    categories = {
        "inclusive": [passthrough],
    },

    weights = {
        "common": {
            "inclusive": ["sf_mu_iso"],
            "bycategory" : {
            }
        },
        "bysample": {
        }
    },

    variations = {
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory" : {
                }
            },
        "bysample": {
        }    
        },
    },

    
   variables= { },

   columns= {
       "common":{
           "inclusive": [ColOut(collection="events", columns=["event", "run", "luminosityBlock"])]
       }
   }
)

