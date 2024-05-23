
#This config.py studies the data SM centered with the list of Wilson coefficient 2


from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import eft_weights_oldv
#import eft_weights

import workflow_W2_SMc
from workflow_W2_SMc import BaseProcessorGen

import custom_cuts


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
            "samples": ["ttHTobb_p1j_EFTcenter_5F_tbarlnutqq_fixscale"],
            "samples_exclude" : [],
            #"year": []
        },
        "subsamples": {}
    },

    workflow = BaseProcessorGen,
    
    skim = [],
    
    preselections = [passthrough],
    categories = {
        "sm": [custom_cuts.cut_events], #SM only inclusive, no weights
        "weight1_5": [custom_cuts.cut_events],
        "weight1_10": [custom_cuts.cut_events],
        "weight2_5": [custom_cuts.cut_events],
        "weight2_10": [custom_cuts.cut_events],
        "weight3_5": [custom_cuts.cut_events],
        "weight3_10": [custom_cuts.cut_events],
        "weight4_5": [custom_cuts.cut_events],
        "weight4_10": [custom_cuts.cut_events],
        "weight5_5": [custom_cuts.cut_events],
        "weight5_10": [custom_cuts.cut_events],
        "weight6_5": [custom_cuts.cut_events],
        "weight6_10": [custom_cuts.cut_events],
        "weight7_5": [custom_cuts.cut_events],
        "weight7_10": [custom_cuts.cut_events],
        "weight8_5": [custom_cuts.cut_events],
        "weight8_10": [custom_cuts.cut_events],
        

        
    },

    weights= {
        "common": {
            "inclusive": [ "genWeight", "XS",], #weights applied to all category
            "bycategory": {
                "sm": [eft_weights_oldv.getSMEFTweight(0)],
                "weight1_5": [eft_weights_oldv.getSMEFTweight(5)],#cthre 5
                "weight1_10": [eft_weights_oldv.getSMEFTweight(6)],#cthre 10
                "weight2_5": [eft_weights_oldv.getSMEFTweight(11)],#ctwre 5
                "weight2_10": [eft_weights_oldv.getSMEFTweight(12)],#ctwre 10
                "weight3_5": [eft_weights_oldv.getSMEFTweight(17)],#ctbre 5
                "weight3_10": [eft_weights_oldv.getSMEFTweight(18)],#ctbre 10
                "weight4_5": [eft_weights_oldv.getSMEFTweight(23)],#cbwre 5
                "weight4_10": [eft_weights_oldv.getSMEFTweight(24)],#cbwre 10
                "weight5_5": [eft_weights_oldv.getSMEFTweight(27)],#chq1 5
                "weight5_10": [eft_weights_oldv.getSMEFTweight(30)],#chq1 10
                "weight6_5": [eft_weights_oldv.getSMEFTweight(35)],#chq3 5
                "weight6_10": [eft_weights_oldv.getSMEFTweight(36)],#chq3 10 
                "weight7_5": [eft_weights_oldv.getSMEFTweight(41)],#cht 5 
                "weight7_10": [eft_weights_oldv.getSMEFTweight(42)],#cht 10 
                "weight8_5": [eft_weights_oldv.getSMEFTweight(47)],#chtbre 5
                "weight8_10": [eft_weights_oldv.getSMEFTweight(48)],#chtbre 10 
                
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
        

        "LHE_w1" : HistConf(
            [Axis(coll="events", field="cthre_BSMc", bins=300, start=0, stop=10, label="$w_1$")], only_categories=['sm'],
        ),

        "LHE_w2" : HistConf(
            [Axis(coll="events", field="ctwre_BSMc", bins=300, start=0, stop=10, label="$w_2$")], only_categories=['sm'],
        ),

        "LHE_w3" : HistConf(
            [Axis(coll="events", field="ctbre_BSMc", bins=300, start=0, stop=10, label="$w_3$")], only_categories=['sm'],
        ),

        "LHE_w4" : HistConf(
            [Axis(coll="events", field="cbwre_BSMc", bins=300, start=0, stop=10, label="$w_4$")], only_categories=['sm'],
        ),

        "LHE_w5" : HistConf(
            [Axis(coll="events", field="chq1_BSMc", bins=300, start=0, stop=10, label="$w_5$")], only_categories=['sm'],
        ),

        "LHE_w6" : HistConf(
            [Axis(coll="events", field="chq3_BSMc", bins=300, start=0, stop=10, label="$w_6$")], only_categories=['sm'],
        ),

        "LHE_w7" : HistConf(
            [Axis(coll="events", field="cht_BSMc", bins=300, start=0, stop=10, label="$w_7$")], only_categories=['sm'],
        ),

        "LHE_w8" : HistConf(
            [Axis(coll="events", field="chtbre_BSMc", bins=300, start=0, stop=10, label="$w_8$")], only_categories=['sm'],
        ),


        "LHE_w1_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="cthre_BSMc", bins=200, start=0, stop=100, label="$w_1$"),
            ], only_categories=['sm'],
        ),
        
         "LHE_w2_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="ctwre_BSMc", bins=200, start=0, stop=100, label="$w_2$"),
            ],only_categories=['sm'],
        ),

         "LHE_w3_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="ctbre_BSMc", bins=200, start=0, stop=100, label="$w_3$"),
            ], only_categories=['sm'],
        ),

         "LHE_w4_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="cbwre_BSMc", bins=200, start=0, stop=100, label="$w_4$"),
            ], only_categories=['sm'],
        ),

         "LHE_w5_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="chq1_BSMc", bins=200, start=0, stop=100, label="$w_5$"),
            ], only_categories=['sm'],
        ),

         "LHE_w6_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="chq3_BSMc", bins=200, start=0, stop=100, label="$w_6$"),
            ], only_categories=['sm'],
        ),

         "LHE_w7_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="cht_BSMc", bins=200, start=0, stop=120, label="$w_7$"),
            ], only_categories=['sm'],
        ),

         "LHE_w8_2d" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=200, start=0, stop=900, label="$H_{p_T}$"),
                Axis(coll="events", field="chtbre_BSMc", bins=200, start=0, stop=100, label="$w_8$"),
            ],only_categories=['sm'],
        ),




         "LHE_w1_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=100, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="cthre_BSMc", bins=100, start=0, stop=10, label="$w_1$"),
            ], only_categories=['sm'],
        ),

         "LHE_w2_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=300, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="ctwre_BSMc", bins=300, start=0, stop=100, label="$w_2$"),
            ], only_categories=['sm'],
        ),

         "LHE_w3_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=300, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="ctbre_BSMc", bins=300, start=0, stop=100, label="$w_3$"),
            ], only_categories=['sm'],
        ),

         "LHE_w4_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=300, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="cbwre_BSMc", bins=300, start=0, stop=100, label="$w_4$"),
            ], only_categories=['sm'],
        ),

         "LHE_w5_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=300, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="chq1_BSMc", bins=300, start=0, stop=60, label="$w_5$"),
            ], only_categories=['sm'],
        ),

         "LHE_w6_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=300, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="chq3_BSMc", bins=300, start=0, stop=100, label="$w_6$"),
            ], only_categories=['sm'],
        ),

         "LHE_w7_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=300, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="cht_BSMc", bins=300, start=0, stop=60, label="$w_7$"),
            ], only_categories=['sm'],
        ),

         "LHE_w8_2d_zoom" : HistConf(
            [
                Axis(coll="HiggsParton", field="pt", bins=300, start=0, stop=700, label="$H_{p_T}$"),
                Axis(coll="events", field="chtbre_BSMc", bins=300, start=0, stop=40, label="$w_8$"),
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
cloudpickle.register_pickle_by_value(workflow_W2_SMc)
cloudpickle.register_pickle_by_value(eft_weights_oldv)
cloudpickle.register_pickle_by_value(custom_cuts)
