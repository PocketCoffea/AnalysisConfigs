
#This config.py studies the data EFT centered with the list of Wilson coefficient 2, in all the points of the Wilson space
#Here you study only the variables

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import eft_weights

import workflow_W2
from workflow_W2 import BaseProcessorGen

import custom_cuts
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

my_weights={}
my_categories={}

my_weights.update({'sm':[eft_weights.getSMEFTweight([0.,0.,0.,0.,0.,0.,0.,0.])]})
my_categories.update({'sm':[custom_cuts.cut_events]})

#weights_value=np.linspace(-20.,20.,41)
weights_value=[5.,10.]

for i in weights_value:

    name='weight1'+'_'+str(int(i))
    my_weights.update({name:[eft_weights.getSMEFTweight([i,0.,0.,0.,0.,0.,0.,0.])]})
    my_categories.update({name:[custom_cuts.cut_events]})

    name='weight2'+'_'+str(int(i))
    my_weights.update({name:[eft_weights.getSMEFTweight([0.,i,0.,0.,0.,0.,0.,0.])]})
    my_categories.update({name:[custom_cuts.cut_events]})

    name='weight3'+'_'+str(int(i))
    my_weights.update({name:[eft_weights.getSMEFTweight([0.,0.,i,0.,0.,0.,0.,0.])]})
    my_categories.update({name:[custom_cuts.cut_events]})

    name='weight4'+'_'+str(int(i))
    my_weights.update({name:[eft_weights.getSMEFTweight([0.,0.,0.,i,0.,0.,0.,0.])]})
    my_categories.update({name:[custom_cuts.cut_events]})

    name='weight5'+'_'+str(int(i))
    my_weights.update({name:[eft_weights.getSMEFTweight([0.,0.,0.,0.,i,0.,0.,0.])]})
    my_categories.update({name:[custom_cuts.cut_events]})

    name='weight6'+'_'+str(int(i))
    my_weights.update({name:[eft_weights.getSMEFTweight([0.,0.,0.,0.,0.,i,0.,0.])]})
    my_categories.update({name:[custom_cuts.cut_events]})

    name='weight7'+'_'+str(int(i))
    my_weights.update({name:[eft_weights.getSMEFTweight([0.,0.,0.,0.,0.,0.,i,0.])]})
    my_categories.update({name:[custom_cuts.cut_events]})

    name='weight8'+'_'+str(int(i))
    my_weights.update({name:[eft_weights.getSMEFTweight([0.,0.,0.,0.,0.,0.,0.,i])]})
    my_categories.update({name:[custom_cuts.cut_events]})



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
    
    categories = my_categories,

    weights= { 
        "common": { 
            "inclusive": [ "genWeight", "XS",], #weights applied to all category
            "bycategory": my_weights,
            
        },
        "bysample": {},
    },
    variations = {
        "weights": {"common": {"inclusive": [], "bycategory": {}}, "bysample": {}},
    }, 
    
    variables = {

        "nGenJet" : HistConf(
            [Axis(coll="events", field="nGenJet", bins=40, start=0, stop=40,
                  label="$N_{j}$")],
        ),
       
        "higgs_pt" : HistConf( 
            [Axis(coll="HiggsParton", field="pt", bins=50, start=0, stop=500,
                  label="$H_{pT}$")],
        ),

        "higgs_eta" : HistConf(
            [Axis(coll="HiggsParton", field="eta", bins=50, start=-3.14, stop=3.14,
                  label="$H_{\eta}$")],
        ),

        "higgs_phi" : HistConf(
            [Axis(coll="HiggsParton", field="phi", bins=50, start=-3.14, stop=3.14,
                  label="$H_{\phi}$")],
        ),


        "deltaR_ht" : HistConf(
            [Axis(coll="events", field="deltaR_ht", bins=50, start=0, stop=5,
                  label="$\Delta R_{ht}$")],
        ),

        "deltaR_hat" : HistConf(
            [Axis(coll="events", field="deltaR_hat", bins=50, start=0, stop=5,
                  label="$\Delta R_{hat}$")],
        ),

        "deltaPhi_tt" : HistConf(
            [Axis(coll="events", field="deltaPhi_tt", bins=50, start=-3.14, stop=3.14,
                  label="$\Delta \phi_{tt}$")],
        ),

        "deltaEta_tt" : HistConf(
            [Axis(coll="events", field="deltaEta_tt", bins=50, start=0, stop=5,
                  label="$\Delta \eta_{tt}$")],
        ),


        "deltaPt_tt" : HistConf(
            [Axis(coll="events", field="deltaPt_tt", bins=50, start=0, stop=600,
                  label="$\Delta pT_{tt}$")],
        ),


         "sumPt_tt" : HistConf(
            [Axis(coll="events", field="sumPt_tt", bins=50, start=0, stop=500,
                  label="$pT_{t}+pT_{at}$")],
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
