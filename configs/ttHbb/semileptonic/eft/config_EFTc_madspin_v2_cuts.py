
#This config.py studies the data BSM centered with the list of Wilson coefficient 2, this is the cut analysis
#I make only SM histograms of the various pt in order to study the cuts


from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import eft_weights
from pocket_coffea.lib.cut_definition import Cut  

import workflow_W2
from workflow_W2 import BaseProcessorGen

import custom_cuts
from custom_cuts import cut_ev
import numpy as np

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection_semileptonic.yaml",
                                                  f"{localdir}/params/eft_params.yaml",
                                                  update=True)




cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/datasets/ttHTobb_EFTcenter_nfeci.json",
                  ],
        "filter" : {   
            "samples": ["ttHTobb_p1j_EFTcenter_5F_tbarqqtlnu"],
            "samples_exclude" : [],
            #"year": []
        },
        "subsamples": {}
    },

    workflow = BaseProcessorGen,
    skim = [],
    
    preselections = [passthrough],
    
    categories = {

        "sm":[passthrough],

        },

    weights= { 
        "common": { 
            "inclusive": [ "genWeight", "XS",], #weights applied to all category
            "bycategory": {

            "sm":[eft_weights.getSMEFTweight([0.,0.,0.,0.,0.,0.,0.,0.])]

            },
            
        },
        "bysample": {},
    },
    variations = {
        "weights": {"common": {"inclusive": [], "bycategory": {}}, "bysample": {}},
    }, 
    
    variables = {

       
        "jet_pt" : HistConf( 
            [Axis(coll="jet_part", field="pt", bins=100, start=0, stop=500,
                  label="$p_T$")],
        ),

        "lep_pt" : HistConf( 
            [Axis(coll="lep_part", field="pt", bins=100, start=0, stop=500,
                  label="$p_T$")],
        ),

        "q_pt" : HistConf( 
            [Axis(coll="q_part", field="pt", bins=100, start=0, stop=500,
                  label="$p_T$")],
        ),

        "g_pt" : HistConf( 
            [Axis(coll="g_part", field="pt", bins=100, start=0, stop=500,
                  label="$p_T$")],
        ),

        "e_pt" : HistConf( 
            [Axis(coll="e_part", field="pt", bins=100, start=0, stop=500,
                  label="$p_T$")],
        ),

        "mu_pt" : HistConf( 
            [Axis(coll="mu_part", field="pt", bins=100, start=0, stop=500,
                  label="$p_T$")],
        ),

        "tau_pt" : HistConf( 
            [Axis(coll="tau_part", field="pt", bins=100, start=0, stop=500,
                  label="$p_T$")],
        ),
         
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
cloudpickle.register_pickle_by_value(workflow_W2)
cloudpickle.register_pickle_by_value(eft_weights)
cloudpickle.register_pickle_by_value(custom_cuts)
