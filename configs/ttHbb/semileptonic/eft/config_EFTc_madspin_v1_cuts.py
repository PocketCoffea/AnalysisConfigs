
#This config.py studies the data BSM centered with the list of Wilson coefficient 2


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


pt_cuts=np.linspace(0,60,31)

cuts_l={}
cuts_j={}

#dictionary with different initial lepton cuts

for l_pt_min in pt_cuts:
     cuts_l[l_pt_min] = Cut(
          name= "",
          params = {
             "j_pt_min":15.,
             "j_eta_max":0.,
             "l_pt_min":l_pt_min,
             "l_eta_max":3.,
          },
          function=cut_ev
     )


for j_pt_min in pt_cuts:
     cuts_j[j_pt_min] = Cut(
          name= "",
          params = {
             "j_pt_min":j_pt_min,
             "j_eta_max":5.,
             "l_pt_min":15.,
             "l_eta_max":3.,
          },
          function=cut_ev
     )




sm_weights_cuts={}
sm_categories_cuts={}

for i in pt_cuts:

    name_l='l_cut_'+str(int(i))
    name_j='j_cut_'+str(int(i))

    sm_weights_cuts.update({name_l:[eft_weights.getSMEFTweight([0.,0.,0.,0.,0.,0.,0.,0.])]})
    sm_categories_cuts.update({name_l:[cuts_l[i]]})

    sm_weights_cuts.update({name_j:[eft_weights.getSMEFTweight([0.,0.,0.,0.,0.,0.,0.,0.])]})
    sm_categories_cuts.update({name_j:[cuts_j[i]]})





cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/datasets/ttHTobb_EFTcenter_nfeci.json",
                  ],
        "filter" : {   
            "samples": ["ttHTobb_p1j_EFTcenter_5F_tbarlnutqq_fixscale"],
            "samples_exclude" : [],
            #"year": []
        },
        "subsamples": {}
    },

    workflow = BaseProcessorGen,
    skim = [],
    
    preselections = [passthrough],
    
    categories = sm_categories_cuts,

    weights= { 
        "common": { 
            "inclusive": [ "genWeight", "XS",], #weights applied to all category
            "bycategory": sm_weights_cuts,
            
        },
        "bysample": {},
    },
    variations = {
        "weights": {"common": {"inclusive": [], "bycategory": {}}, "bysample": {}},
    }, 
    
    variables = {

       
        "higgs_pt" : HistConf( 
            [Axis(coll="HiggsParton", field="pt", bins=50, start=0, stop=500,
                  label="$H_{pT}$")],
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
