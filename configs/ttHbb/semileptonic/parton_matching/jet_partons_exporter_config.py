from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel, get_nBtagMin
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *

import workflow
from workflow import PartonMatchingProcessor

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
                                                  f"{localdir}/params/object_preselection.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  update=True)

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [
            f"{localdir}/datasets/signal_ttHTobb_forTraining_local.json",
            # f"{localdir}/datasets/signal_ttHTobb_local.json",
            # f"{localdir}/datasets/backgrounds_MC_ttbar_local.json",
            # f"{localdir}/datasets/backgrounds_MC_TTbb_local.json"
        ],
        "filter" : {
            "samples": ["ttHTobb", "TTToSemiLeptonic", "TTbbSemiLeptonic"],
            "samples_exclude" : [],
            "year": ["2016_PreVFP",
                     "2016_PostVFP",
                     "2017","2018"] #All the years
        }
    },

    workflow = PartonMatchingProcessor,
    workflow_options = {"parton_jet_min_dR": 0.3,
                        "dump_columns_as_arrays_per_chunk": "root://t3se01.psi.ch:1094//store/user/dvalsecc/ttHbb/output_columns_parton_matching/sig_bkg_30_08_2023_v2/"},
    
    skim = [get_nObj_min(4, 15., coll="Jet"),
            get_nBtagMin(3, 15., coll="Jet"),
            get_HLTsel()], 
    
    preselections = [semileptonic_presel],
    categories = {
        "semilep_LHE": [semilep_lhe]
    },

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
    columns = {
        "common": {
            "bycategory": {
                    "semilep_LHE": [
                        ColOut("Parton", ["pt", "eta", "phi", "mass", "pdgId", "provenance"], flatten=False),
                        ColOut(
                            "PartonMatched",
                            ["pt", "eta", "phi","mass", "pdgId", "provenance", "dRMatchedJet",], flatten=False
                        ),
                        ColOut(
                            "JetGood",
                            ["pt", "eta", "phi", "hadronFlavour", "btagDeepFlavB"], flatten=False
                        ),
                        ColOut(
                            "JetGoodMatched",
                            [
                                "pt",
                                "eta",
                                "phi",
                                "hadronFlavour",
                                "btagDeepFlavB",
                                "dRMatchedJet",
                            ], flatten=False
                        ),
                        
                        ColOut("LeptonGood",
                               ["pt","eta","phi"],flatten=False,
                               pos_end=1, store_size=False),
                        ColOut("MET", ["phi","pt","significance"], flatten=False),
                        ColOut("Generator",["x1","x2","id1","id2","xpdf1","xpdf2"], flatten=False),
                        ColOut("LeptonParton",["pt","eta","phi","mass","pdgId"], flatten=False)
                    ]
                }
        },
        "bysample":{
            "ttHTobb":{
                "bycategory": {
                    "semilep_LHE": [ColOut("HiggsParton",
                                           ["pt","eta","phi","mass","pdgId"], pos_end=1, store_size=False, flatten=False)]
                }
            }
        }
    },
)




run_options = {
        "executor"       : "dask/slurm",
        "env"            : "conda",
        "workers"        : 1,
        "scaleout"       : 100,
        "worker_image"   : "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-cc7-lat   est",
        "queue"          : "standard",
        "walltime"       : "03:00:00",
        "mem_per_worker" : "5GB", # GB
        "disk_per_worker" : "1GB", # GB
        "exclusive"      : False,
        "chunk"          : 400000,
        "retries"        : 50,
        "treereduction"  : 20,
        "adapt"          : False,
        
    }
   


if "dask"  in run_options["executor"]:
    import cloudpickle
    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(custom_cut_functions)
    cloudpickle.register_pickle_by_value(custom_cuts)
