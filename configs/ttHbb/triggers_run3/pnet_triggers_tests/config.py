from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel, get_nBtagMin
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *

import workflow
from workflow import TriggerProcessor

import cuts
from cuts import *
import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/triggers.yaml",
                                                  f"{localdir}/params/general_params.yaml",
                                                  update=True)


cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/ttHbb_run3_privateprod.json"],
        "filter" : {
            "samples": ["ttHTobb"],
            "samples_exclude" : [],
            "year": ["Run3"]
        }
    },

    workflow = TriggerProcessor,
    workflow_options = {},
    
    skim = [get_nObj_min(4, 20., coll="Jet")], 
    
    preselections = [],
    categories = {
        "inclusive": [passthrough],
        "semilep": [semilep_lhe, notau, get_nObj_min(1, 20., coll="Lepton")],
        "semilep_singlelepHLT" : [semilep_lhe, notau, get_nObj_min(1, 20., coll="Lepton"), get_HLTsel(["SingleLepton"])],
        "semilep_tripleB+singleLep": [semilep_lhe, notau,get_nObj_min(1, 20., coll="Lepton"), get_HLTsel(["TripleBTag","SingleLepton"])] ,
        "semilep_doubleB_tighter+singleLep": [semilep_lhe, notau,get_nObj_min(1, 20., coll="Lepton"), get_HLTsel(["Jet4_btag2_tighter","SingleLepton"])] ,
        "semilep_doubleB_looser+singleLep": [semilep_lhe, notau,get_nObj_min(1, 20., coll="Lepton"), get_HLTsel(["Jet4_btag2_looser","SingleLepton"])],
        "semilep_tripleB": [semilep_lhe, notau,get_nObj_min(1, 20., coll="Lepton"), get_HLTsel(["TripleBTag"])] ,
        "semilep_doubleB_tighter": [semilep_lhe, notau,get_nObj_min(1, 20., coll="Lepton"), get_HLTsel(["Jet4_btag2_tighter"])] ,
        "semilep_doubleB_looser": [semilep_lhe, notau,get_nObj_min(1, 20., coll="Lepton"), get_HLTsel(["Jet4_btag2_looser"])],
  
        "had": [had_lhe  ],
        "had_tripleB": [had_lhe, get_HLTsel(["TripleBTag"])] ,
        "had_doubleB_tighter": [had_lhe, get_HLTsel(["Jet4_btag2_tighter"])] ,
        "had_doubleB_looser": [had_lhe, get_HLTsel(["Jet4_btag2_looser"])]
    },
    

    weights= {
        "common": {
            "inclusive": [
                "genWeight", "lumi","XS",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations = {
        "weights": {"common": {"inclusive": [], "bycategory": {}},
                    "bysample": {}},
    },

    variables = {
        "higgs_pt": HistConf([
            Axis(coll="higgs", field="pt",
                 bins=30, start=0, stop=400, label="Higgs $p_T$")
        ]),
        "HT": HistConf([
            Axis(coll="LHE", field="HT",
                 bins=40, start=0, stop=1500, label="$H_T$")
        ]),
        "Lepton_pt":  HistConf([
            Axis(coll="Lepton", field="pt", pos=0,
                 bins=40, start=0, stop=300, label="Lepton $p_T$")
        ]),
    }

)


run_options = {
        "executor"       : "dask/lxplus",
        "env"            : "singularity",
        "workers"        : 1,
        "scaleout"       : 10,
        "worker_image"   : "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-cc7-latest",
        "queue"          : "espresso",
        "walltime"       : "00:40:00",
        "mem_per_worker" : "2GB", # GB
        "disk_per_worker" : "1GB", # GB
        "exclusive"      : False,
        "chunk"          : 5000,
        "retries"        : 50,
        "treereduction"  : 20,
        "adapt"          : False,
        
    }
   


if "dask"  in run_options["executor"]:
    import cloudpickle
    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(cuts)


    
