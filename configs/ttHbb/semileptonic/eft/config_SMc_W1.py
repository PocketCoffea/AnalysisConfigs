
#This config.py studies the data SM centered with the list of Wilson coefficient 1


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
        "jsons": [f"{localdir}/datasets/signal_ttHTobb_T2.json",
                  ],
        "filter" : {   
            "samples": ["ttHTobb_SMcenter_EFT", "ttHTobb_SMcenter_p1j_EFT" ],#,"ttHTobb_BSM"],
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
        "weight17_applied": [passthrough],
        "weight18_applied": [passthrough],
        #"weight19_applied": [passthrough],
        #"weight20_applied": [passthrough],
        #"weight21_applied": [passthrough],
        #"weight22_applied": [passthrough],
        #"weight23_applied": [passthrough],
        #"weight24_applied": [passthrough],
        #"weight25_applied": [passthrough],
        #"weight26_applied": [passthrough],
        #"weight27_applied": [passthrough],
        #"weight28_applied": [passthrough],
        #"weight29_applied": [passthrough],
        #"weight30_applied": [passthrough],
        #"weight31_applied": [passthrough],
        #"weight32_applied": [passthrough],
        #"weight33_applied": [passthrough],
        #"weight34_applied": [passthrough],
        #"weight35_applied": [passthrough],
        #"weight36_applied": [passthrough],
        #"weight37_applied": [passthrough],
        #"weight38_applied": [passthrough],
        #"weight39_applied": [passthrough],
        #"weight40_applied": [passthrough],
        #"weight41_applied": [passthrough],
        #"weight42_applied": [passthrough],
        #"weight43_applied": [passthrough],
        #"weight44_applied": [passthrough],
        #"weight45_applied": [passthrough],
        #"weight46_applied": [passthrough],
        #"weight47_applied": [passthrough],
        #"weight48_applied": [passthrough],
        #"weight49_applied": [passthrough],
        #"weight50_applied": [passthrough],
        #"weight51_applied": [passthrough],
        #"weight52_applied": [passthrough],
        #"weight53_applied": [passthrough],
        #"weight54_applied": [passthrough],

        #cut applied with respect to ctwre > maxweight
        "cut_ctwre_sm": [custom_cuts.cut_ctwre],
        "cut_ctwre_w_cthre": [custom_cuts.cut_ctwre],
        "cut_ctwre_w_cht": [custom_cuts.cut_ctwre],
        "cut_ctwre_w_ctwre": [custom_cuts.cut_ctwre],
        "cut_ctwre_w_cbwre": [custom_cuts.cut_ctwre],
        "cut_ctwre_w_ctbre": [custom_cuts.cut_ctwre],

        #cut applied with respect to cbwre > maxweight
        "cut_cbwre_sm": [custom_cuts.cut_cbwre],       
        "cut_cbwre_w_cthre": [custom_cuts.cut_cbwre],
        "cut_cbwre_w_cht": [custom_cuts.cut_cbwre],
        "cut_cbwre_w_ctwre": [custom_cuts.cut_cbwre],
        "cut_cbwre_w_cbwre": [custom_cuts.cut_cbwre],
        "cut_cbwre_w_ctbre": [custom_cuts.cut_cbwre],

        #cut applied with respect to ctbre > maxweight
        "cut_ctbre_sm": [custom_cuts.cut_ctbre],       
        "cut_ctbre_w_cthre": [custom_cuts.cut_ctbre],
        "cut_ctbre_w_cht": [custom_cuts.cut_ctbre],
        "cut_ctbre_w_ctwre": [custom_cuts.cut_ctbre],
        "cut_ctbre_w_cbwre": [custom_cuts.cut_ctbre],
        "cut_ctbre_w_ctbre": [custom_cuts.cut_ctbre],

        
    },

    weights= {
        "common": {
            "inclusive": [ "genWeight", "XS",], #weights applied to all category
            "bycategory": {
                "weight1_applied": [eft_weights.getSMEFTweight(0)],#cthre 0.5
                "weight2_applied": [eft_weights.getSMEFTweight(1)],#cthre 1
                "weight3_applied": [eft_weights.getSMEFTweight(2)],#cuhre 0.5
                "weight4_applied": [eft_weights.getSMEFTweight(3)],#cuhre 1
                "weight5_applied": [eft_weights.getSMEFTweight(4)],#cdhre 0.5
                "weight6_applied": [eft_weights.getSMEFTweight(5)],#cdhre 1
                "weight7_applied": [eft_weights.getSMEFTweight(6)],#cbhre 0.5
                "weight8_applied": [eft_weights.getSMEFTweight(7)],#cbhre 1
                "weight9_applied": [eft_weights.getSMEFTweight(8)],#cht 0.5
                "weight10_applied": [eft_weights.getSMEFTweight(9)],#cht 1
                "weight11_applied": [eft_weights.getSMEFTweight(10)],#ctwre 0.5
                "weight12_applied": [eft_weights.getSMEFTweight(11)],#ctwre 1 
                "weight13_applied": [eft_weights.getSMEFTweight(12)],#cbwre 0.5 
                "weight14_applied": [eft_weights.getSMEFTweight(13)],#cbwre 1 
                "weight15_applied": [eft_weights.getSMEFTweight(14)],#cw 0.5
                "weight16_applied": [eft_weights.getSMEFTweight(15)],#cw 1 
                "weight17_applied": [eft_weights.getSMEFTweight(16)],#ctbre 0.5  
                "weight18_applied": [eft_weights.getSMEFTweight(17)],#ctbre 1 

              #  "weight19_applied": [eft_weights.getSMEFTweight(18)],#mixed weights
              #  "weight20_applied": [eft_weights.getSMEFTweight(19)],  
              #  "weight21_applied": [eft_weights.getSMEFTweight(20)],  
              #  "weight22_applied": [eft_weights.getSMEFTweight(21)],  
              #  "weight23_applied": [eft_weights.getSMEFTweight(22)],  
              #  "weight24_applied": [eft_weights.getSMEFTweight(23)],  
              #  "weight25_applied": [eft_weights.getSMEFTweight(24)],  
              #  "weight26_applied": [eft_weights.getSMEFTweight(25)],  
              #  "weight27_applied": [eft_weights.getSMEFTweight(26)],  
              #  "weight28_applied": [eft_weights.getSMEFTweight(27)],  
              #  "weight29_applied": [eft_weights.getSMEFTweight(28)],  
              #  "weight30_applied": [eft_weights.getSMEFTweight(29)],  
              #  "weight31_applied": [eft_weights.getSMEFTweight(30)],  
              #  "weight32_applied": [eft_weights.getSMEFTweight(31)],  
              #  "weight33_applied": [eft_weights.getSMEFTweight(32)],  
              #  "weight34_applied": [eft_weights.getSMEFTweight(33)],  
              #  "weight35_applied": [eft_weights.getSMEFTweight(34)],  
              #  "weight36_applied": [eft_weights.getSMEFTweight(35)],  
              #  "weight37_applied": [eft_weights.getSMEFTweight(36)],  
              #  "weight38_applied": [eft_weights.getSMEFTweight(37)],  
              #  "weight39_applied": [eft_weights.getSMEFTweight(38)],  
              #  "weight40_applied": [eft_weights.getSMEFTweight(39)],  
              #  "weight41_applied": [eft_weights.getSMEFTweight(40)],  
              #  "weight42_applied": [eft_weights.getSMEFTweight(41)],  
              #  "weight43_applied": [eft_weights.getSMEFTweight(42)],  
              #  "weight44_applied": [eft_weights.getSMEFTweight(43)],  
              #  "weight45_applied": [eft_weights.getSMEFTweight(44)],  
              #  "weight46_applied": [eft_weights.getSMEFTweight(45)],  
              #  "weight47_applied": [eft_weights.getSMEFTweight(46)],  
              #  "weight48_applied": [eft_weights.getSMEFTweight(47)],  
              #  "weight49_applied": [eft_weights.getSMEFTweight(48)],  
              #  "weight50_applied": [eft_weights.getSMEFTweight(49)],  
              #  "weight51_applied": [eft_weights.getSMEFTweight(50)],  
              #  "weight52_applied": [eft_weights.getSMEFTweight(51)],  
              #  "weight53_applied": [eft_weights.getSMEFTweight(52)],  
              #  "weight54_applied": [eft_weights.getSMEFTweight(53)],  
         

        #the weight applied is the one after w_ 
        "cut_ctwre_w_cthre": [eft_weights.getSMEFTweight(1)],
        "cut_ctwre_w_cht": [eft_weights.getSMEFTweight(9)],
        "cut_ctwre_w_ctwre": [eft_weights.getSMEFTweight(11)],
        "cut_ctwre_w_cbwre": [eft_weights.getSMEFTweight(13)],
        "cut_ctwre_w_ctbre": [eft_weights.getSMEFTweight(17)],

        #cut applied with respect to cbwre > maxweight
        "cut_cbwre_w_cthre": [eft_weights.getSMEFTweight(1)],
        "cut_cbwre_w_cht": [eft_weights.getSMEFTweight(9)],
        "cut_cbwre_w_ctwre": [eft_weights.getSMEFTweight(11)],
        "cut_cbwre_w_cbwre": [eft_weights.getSMEFTweight(13)],
        "cut_cbwre_w_ctbre": [eft_weights.getSMEFTweight(17)],

        #cut applied with respect to ctbre > maxweight
        "cut_ctbre_w_cthre": [eft_weights.getSMEFTweight(1)],
        "cut_ctbre_w_cht": [eft_weights.getSMEFTweight(9)],
        "cut_ctbre_w_ctwre": [eft_weights.getSMEFTweight(11)],
        "cut_ctbre_w_cbwre": [eft_weights.getSMEFTweight(13)],
        "cut_ctbre_w_ctbre": [eft_weights.getSMEFTweight(17)],
                      
                

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
        
'''
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


'''
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
