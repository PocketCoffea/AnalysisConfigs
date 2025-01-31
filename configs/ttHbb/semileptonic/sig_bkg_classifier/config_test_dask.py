from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel, get_nBtagMin
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.lib.weights.common.weights_run2_UL import SF_ele_trigger
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *

import workflow_test_spanet
from workflow_test_spanet import TestDaskProcessor

import configs.ttHbb.semileptonic.common.cuts.custom_cut_functions as custom_cut_functions
import configs.ttHbb.semileptonic.common.cuts.custom_cuts as custom_cuts
from configs.ttHbb.semileptonic.common.cuts.custom_cut_functions import *
from configs.ttHbb.semileptonic.common.cuts.custom_cuts import *
import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection_semileptonic.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  f"{localdir}/params/lepton_scale_factors.yaml",
                                                  f"{localdir}/params/plotting_style.yaml",
                                                  update=True)

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/datasets/signal_ttHTobb_local.json",
                  f"{localdir}/datasets/signal_ttHTobb_ttToSemiLep_local.json",
                  f"{localdir}/datasets/backgrounds_MC_TTbb_local.json",
                  f"{localdir}/datasets/backgrounds_MC_ttbar_local.json"
                  ],
        "filter" : {
            "samples": ["ttHTobb",
                        "ttHTobb_ttToSemiLep",
                        "TTbbSemiLeptonic",
                        "TTToSemiLeptonic"
                        ],
            "samples_exclude" : [],
            "year": ["2016_PreVFP",
                     "2016_PostVFP",
                     "2017",
                     "2018"
                     ] #All the years
        },
        "subsamples": {
            'TTbbSemiLeptonic' : {
                'TTbbSemiLeptonic_tt+LF'   : [get_genTtbarId_100_eq(0)],
                'TTbbSemiLeptonic_tt+C'    : [get_genTtbarId_100_eq([41, 42, 43, 44, 45, 46])],
                'TTbbSemiLeptonic_tt+B'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56])],
            },
            'TTToSemiLeptonic' : {
                'TTToSemiLeptonic_tt+LF'   : [get_genTtbarId_100_eq(0)],
                'TTToSemiLeptonic_tt+C'    : [get_genTtbarId_100_eq([41, 42, 43, 44, 45, 46])],
                'TTToSemiLeptonic_tt+B'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56])],
            },
        }
    },

    workflow = TestDaskProcessor,
    
    skim = [get_nObj_min(4, 15., "Jet"),
            get_nBtagMin(3, 15., coll="Jet", wp="M"),
            get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon"])],
    
    preselections = [semileptonic_presel],
    categories = {
        "baseline": [passthrough],
        "semilep_LHE": [semilep_lhe]
    },

    weights_classes = common_weights + [SF_ele_trigger],
    weights= {
        "common": {
            "inclusive": [
                "genWeight", "lumi","XS",
                "pileup",
                "sf_ele_reco", "sf_ele_id",
                "sf_mu_id", "sf_mu_iso",
                "sf_btag",
                "sf_jet_puId",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations = {
        "weights": {"common": {"inclusive": [], "bycategory": {}}, "bysample": {}},
    },
    
    variables = {},
)

# Registering custom functions
import cloudpickle
cloudpickle.register_pickle_by_value(workflow_test_spanet)
cloudpickle.register_pickle_by_value(custom_cut_functions)
cloudpickle.register_pickle_by_value(custom_cuts)
