
#This config.py studies the data SM centered with the list of Wilson coefficient 2


from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import eft_weights

import workflow
from workflow import BaseProcessorGen

import custom_cuts


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
        "jsons": [f"{localdir}/datasets/ttHTobb_SMcenter.json",
                  ],
        "filter" : {   
            "samples": ["ttHTobb_p1j_SMcenter"],
            "samples_exclude" : [],
            #"year": []
        },
        "subsamples": {}
    },

    workflow = BaseProcessorGen,
    
    skim = [],
    
    preselections = [passthrough],
    categories = {
        "sm": [passthrough], #SM only inclusive, no weights
        "weight1_applied": [passthrough],
        "weight2_applied": [passthrough],
        "weight3_applied": [passthrough],
        "weight4_applied": [passthrough],
        "weight5_applied": [passthrough],
        "weight6_applied": [passthrough],
        "weight7_applied": [passthrough],
        "weight8_applied": [passthrough],
        "weight9_applied": [passthrough],
        "weight10_applied": [passthrough],
        "weight11_applied": [passthrough],
        "weight12_applied": [passthrough],
        "weight13_applied": [passthrough],
        "weight14_applied": [passthrough],
        "weight15_applied": [passthrough],
        "weight16_applied": [passthrough],
        

        
    },

    weights= {
        "common": {
            "inclusive": [ "genWeight", "XS",], #weights applied to all category
            "bycategory": {
                "weight1_applied": [eft_weights.getSMEFTweight(0)],#cthre 5
                "weight2_applied": [eft_weights.getSMEFTweight(1)],#cthre 10
                "weight3_applied": [eft_weights.getSMEFTweight(2)],#ctwre 5
                "weight4_applied": [eft_weights.getSMEFTweight(3)],#ctwre 10
                "weight5_applied": [eft_weights.getSMEFTweight(4)],#ctbre 5
                "weight6_applied": [eft_weights.getSMEFTweight(5)],#ctbre 10
                "weight7_applied": [eft_weights.getSMEFTweight(6)],#cbwre 5
                "weight8_applied": [eft_weights.getSMEFTweight(7)],#cbwre 10
                "weight9_applied": [eft_weights.getSMEFTweight(8)],#chq1 5
                "weight10_applied": [eft_weights.getSMEFTweight(9)],#chq1 10
                "weight11_applied": [eft_weights.getSMEFTweight(10)],#chq3 5
                "weight12_applied": [eft_weights.getSMEFTweight(11)],#chq3 10 
                "weight13_applied": [eft_weights.getSMEFTweight(12)],#cht 5 
                "weight14_applied": [eft_weights.getSMEFTweight(13)],#cht 10 
                "weight15_applied": [eft_weights.getSMEFTweight(14)],#chtbre 5
                "weight16_applied": [eft_weights.getSMEFTweight(15)],#chtbre 10 
                
            }, #I can specify categories to whom I apply only one weight
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


        "deltaPt_tt" : HistConf(
            [Axis(coll="events", field="deltaPt_tt", bins=50, start=0, stop=600,
                  label="$\Delta pT_{tt}$")],
        ),


         "sumPt_tt" : HistConf(
            [Axis(coll="events", field="sumPt_tt", bins=50, start=0, stop=500,
                  label="$pT_{t}+pT_{at}$")],
        ),
        

        "LHE_w1" : HistConf(
            [Axis(coll="events", field="LHE_w1", bins=300, start=0, stop=5, label="$w_1$")], only_categories=['sm'],
        ),

        "LHE_w2" : HistConf(
            [Axis(coll="events", field="LHE_w2", bins=300, start=0, stop=5, label="$w_2$")], only_categories=['sm'],
        ),

        "LHE_w3" : HistConf(
            [Axis(coll="events", field="LHE_w3", bins=300, start=0, stop=5, label="$w_3$")], only_categories=['sm'],
        ),

        "LHE_w4" : HistConf(
            [Axis(coll="events", field="LHE_w4", bins=300, start=0, stop=5, label="$w_4$")], only_categories=['sm'],
        ),

        "LHE_w5" : HistConf(
            [Axis(coll="events", field="LHE_w5", bins=300, start=0, stop=5, label="$w_5$")], only_categories=['sm'],
        ),

        "LHE_w6" : HistConf(
            [Axis(coll="events", field="LHE_w6", bins=300, start=0, stop=5, label="$w_6$")], only_categories=['sm'],
        ),

        "LHE_w7" : HistConf(
            [Axis(coll="events", field="LHE_w7", bins=300, start=0, stop=5, label="$w_7$")], only_categories=['sm'],
        ),

        "LHE_w8" : HistConf(
            [Axis(coll="events", field="LHE_w8", bins=300, start=0, stop=5, label="$w_8$")], only_categories=['sm'],
        ),

        "LHE_w9" : HistConf(
            [Axis(coll="events", field="LHE_w9", bins=300, start=0, stop=5, label="$w_9$")], only_categories=['sm'],
        ),



        "LHE_w1_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w1", bins=200, start=0, stop=100, label="$w_1$"),
            ], only_categories=['sm','cut_ctwre_sm', 'cut_cbwre_sm','cut_ctbre_sm'],
        ),
        
         "LHE_w2_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w2", bins=200, start=0, stop=100, label="$w_2$"),
            ],only_categories=['sm','cut_ctwre_sm', 'cut_cbwre_sm','cut_ctbre_sm'],
        ),

         "LHE_w3_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w3", bins=200, start=0, stop=100, label="$w_3$"),
            ], only_categories=['sm','cut_ctwre_sm', 'cut_cbwre_sm','cut_ctbre_sm'],
        ),

         "LHE_w4_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w4", bins=200, start=0, stop=100, label="$w_4$"),
            ], only_categories=['sm','cut_ctwre_sm', 'cut_cbwre_sm','cut_ctbre_sm'],
        ),

         "LHE_w5_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w5", bins=200, start=0, stop=100, label="$w_5$"),
            ], only_categories=['sm','cut_ctwre_sm', 'cut_cbwre_sm','cut_ctbre_sm'],
        ),

         "LHE_w6_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w6", bins=200, start=0, stop=100, label="$w_6$"),
            ], only_categories=['sm','cut_ctwre_sm', 'cut_cbwre_sm','cut_ctbre_sm'],
        ),

         "LHE_w7_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w7", bins=200, start=0, stop=120, label="$w_7$"),
            ], only_categories=['sm','cut_ctwre_sm', 'cut_cbwre_sm','cut_ctbre_sm'],
        ),

         "LHE_w8_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w8", bins=200, start=0, stop=100, label="$w_8$"),
            ],only_categories=['sm','cut_ctwre_sm', 'cut_cbwre_sm','cut_ctbre_sm'],
        ),

         "LHE_w9_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w9", bins=200, start=0, stop=100, label="$w_9$"),
            ], only_categories=['sm','cut_ctwre_sm', 'cut_cbwre_sm','cut_ctbre_sm'],
        ),



         "LHE_w1_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=500, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w1", bins=500, start=0.88, stop=0.89, label="$w_1$"),
            ], only_categories=['sm'],
        ),

         "LHE_w2_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=1000, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w2", bins=1000, start=0, stop=10, label="$w_2$"),
            ], only_categories=['sm'],
        ),

         "LHE_w3_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=1000, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w3", bins=1000, start=0, stop=10, label="$w_3$"),
            ], only_categories=['sm'],
        ),

         "LHE_w4_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=1000, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w4", bins=1000, start=0, stop=10, label="$w_4$"),
            ], only_categories=['sm'],
        ),

         "LHE_w5_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=1000, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w5", bins=1000, start=0.8, stop=1.2, label="$w_5$"),
            ], only_categories=['sm'],
        ),

         "LHE_w6_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w6", bins=200, start=0, stop=1.5, label="$w_6$"),
            ], only_categories=['sm'],
        ),

         "LHE_w7_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=100, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w7", bins=100, start=0.98, stop=1.03, label="$w_7$"),
            ], only_categories=['sm'],
        ),

         "LHE_w8_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=1000, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w8", bins=1000, start=0, stop=10, label="$w_8$"),
            ], only_categories=['sm'],
        ),

         "LHE_w9_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="LHE_w9", bins=200, start=0.9999, stop=1.0001, label="$w_9$"),
            ], only_categories=['sm'],
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
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(eft_weights)
cloudpickle.register_pickle_by_value(custom_cuts)
