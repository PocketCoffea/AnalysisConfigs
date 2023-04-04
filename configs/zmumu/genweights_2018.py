from pocket_coffea.parameters.cuts.preselection_cuts import passthrough, semileptonic_presel_nobtag
from pocket_coffea.workflows.genweights import genWeightsProcessor
from configs.zmumu import datasets
datasets_abspath = datasets.__path__[0]

samples = ["DYJetsToLL"]

cfg =  {
    "dataset" : {
        "jsons": [f"{datasets_abspath}/DYJetsToLL_M-50.json"],
        "filter" : {
            "samples": samples,
            "samples_exclude" : [],
            "year": ['2018']
        },
    },

    # Input and output files
    "workflow" : genWeightsProcessor,
    "output"   : "output/genweights/genweights_2018",
    "worflow_options" : {},

    "run_options" : {
        "executor"       : "futures",
        "workers"        : 1,
        "scaleout"       : 15,
        "queue"          : "standard",
        "walltime"       : "12:00:00",
        "mem_per_worker" : "4GB", # GB
        "exclusive"      : False,
        "chunk"          : 400000,
        "retries"        : 50,
        "treereduction"  : 10,
        "max"            : None,
        "skipbadfiles"   : None,
        "voms"           : None,
        "limit"          : None,
        "adapt"          : False,
    },

    # Cuts and plots settings
    "finalstate" : "dimuon",
    "skim": [passthrough],
    "preselections" : [passthrough],
    "categories": {
        "baseline": [passthrough],
    },

    

    "weights": {
        "common": {
            "inclusive": ["genWeight","lumi","XS",
                          ],
            "bycategory" : {

            }
        },
        "bysample": {
        }
    },

    "variations": {
        "weights": {
            "common": {
                "inclusive": [  ],
                "bycategory" : {

                }
            },
        "bysample": {
        }    
        },
        
    },

   "variables": {

   },

}
