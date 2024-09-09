from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.cut_functions import get_nObj_eq, get_nObj_min, get_HLTsel, get_nBtagMin, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *

import workflow, workflow_spanet, workflow_control_regions
from workflow_control_regions import ControlRegionsProcessor

import custom_cut_functions
import custom_cuts
from custom_cut_functions import *
from custom_cuts import *
from params.axis_settings import axis_settings
import quantile_transformer
from quantile_transformer import WeightedQuantileTransformer

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Define SPANet model path for inference
spanet_model_path = "/pnfs/psi.ch/cms/trivcat/store/user/mmarcheg/ttHbb/models/meanloss_multiclassifier_btag_LMH/spanet_output/version_0/spanet.onnx"

# Define tthbb working points for SPANet
tthbb_L = 0.4
tthbb_M = 0.6
ttlf_wp = 0.6

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection_semileptonic.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  f"{localdir}/params/lepton_scale_factors.yaml",
                                                  f"{localdir}/params/plotting_style.yaml",
                                                  f"{localdir}/params/quantile_transformer.yaml",
                                                  update=True)

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/datasets/signal_ttHTobb_local.json",
                  f"{localdir}/datasets/signal_ttHTobb_ttToSemiLep_local.json",
                  f"{localdir}/datasets/backgrounds_MC_TTbb_local.json",
                  f"{localdir}/datasets/backgrounds_MC_ttbar_local.json",
                  f"{localdir}/datasets/backgrounds_MC_local.json",
                  f"{localdir}/datasets/DATA_SingleEle_local.json",
                  f"{localdir}/datasets/DATA_SingleMuon_local.json",
                  ],
        "filter" : {
            "samples": ["ttHTobb",
                        "ttHTobb_ttToSemiLep",
                        "TTbbSemiLeptonic",
                        "TTToSemiLeptonic",
                        "TTTo2L2Nu",
                        "SingleTop",
                        "WJetsToLNu_HT",
                        "DATA_SingleEle",
                        "DATA_SingleMuon"
                        ],
            "samples_exclude" : [],
            "year": [#"2016_PreVFP",
                     #"2016_PostVFP",
                     "2017",
                     #"2018"
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

    workflow = ControlRegionsProcessor,
    workflow_options = {"parton_jet_min_dR": 0.3,
                        "spanet_model": spanet_model_path},
    
    skim = [get_nPVgood(1),
            eventFlags,
            goldenJson,
            get_nObj_min(4, 15., "Jet"),
            get_nBtagMin(3, 15., coll="Jet", wp="M"),
            get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon"])],
    
    preselections = [semileptonic_presel],
    categories = {
        "semilep": [passthrough],
        "CR1": [get_ttlf_max(ttlf_wp), get_CR1(tthbb_L)],
        "CR2": [get_ttlf_max(ttlf_wp), get_CR2(tthbb_L, tthbb_M)],
        "SR": [get_ttlf_max(ttlf_wp), get_SR(tthbb_M)],
        "4jCR1": [get_ttlf_max(ttlf_wp), get_CR1(tthbb_L), get_nObj_eq(4, coll="JetGood")],
        "4jCR2": [get_ttlf_max(ttlf_wp), get_CR2(tthbb_L, tthbb_M), get_nObj_eq(4, coll="JetGood")],
        "4jSR": [get_ttlf_max(ttlf_wp), get_SR(tthbb_M), get_nObj_eq(4, coll="JetGood")],
        "5jCR1": [get_ttlf_max(ttlf_wp), get_CR1(tthbb_L), get_nObj_eq(5, coll="JetGood")],
        "5jCR2": [get_ttlf_max(ttlf_wp), get_CR2(tthbb_L, tthbb_M), get_nObj_eq(5, coll="JetGood")],
        "5jSR": [get_ttlf_max(ttlf_wp), get_SR(tthbb_M), get_nObj_eq(5, coll="JetGood")],
        ">=6jCR1": [get_ttlf_max(ttlf_wp), get_CR1(tthbb_L), get_nObj_min(6, coll="JetGood")],
        ">=6jCR2": [get_ttlf_max(ttlf_wp), get_CR2(tthbb_L, tthbb_M), get_nObj_min(6, coll="JetGood")],
        ">=6jSR": [get_ttlf_max(ttlf_wp), get_SR(tthbb_M), get_nObj_min(6, coll="JetGood")],
    },

    weights= {
        "common": {
            "inclusive": [
                "genWeight", "lumi","XS",
                "pileup",
                "sf_ele_reco", "sf_ele_id", "sf_ele_trigger",
                "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
                "sf_btag",
                "sf_jet_puId",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations = {
        "weights": {
            "common": {
                "inclusive": ["pileup",
                              "sf_ele_reco", "sf_ele_id", "sf_ele_trigger",
                              "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
                              "sf_btag",
                              "sf_jet_puId",
                              ],
                "bycategory": {}
            },
            "bysample": {}
        },
    },
    
    variables = {
        **count_hist(name="nLeptons", coll="LeptonGood",bins=3, start=0, stop=3),
        **count_hist(name="nJets", coll="JetGood",bins=10, start=4, stop=14),
        **count_hist(name="nBJets", coll="BJetGood",bins=10, start=0, stop=10),
        **ele_hists(axis_settings=axis_settings),
        **muon_hists(axis_settings=axis_settings),
        **met_hists(coll="MET", axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=0, axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=1, axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=2, axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=3, axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=4, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=0, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=1, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=2, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=3, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=4, axis_settings=axis_settings),
        "jets_Ht" : HistConf(
          [Axis(coll="events", field="JetGood_Ht", bins=100, start=0, stop=2500,
                label="Jets $H_T$ [GeV]")]
        ),
        "bjets_Ht" : HistConf(
          [Axis(coll="events", field="BJetGood_Ht", bins=100, start=0, stop=2500,
                label="B-Jets $H_T$ [GeV]")]
        ),
        "lightjets_Ht" : HistConf(
          [Axis(coll="events", field="LightJetGood_Ht", bins=100, start=0, stop=2500,
                label="Light-Jets $H_T$ [GeV]")]
        ),
        "deltaRbb_min" : HistConf(
            [Axis(coll="events", field="deltaRbb_min", bins=50, start=0, stop=5,
                  label="$\Delta R_{bb}$")]
        ),
        "deltaEtabb_min" : HistConf(
            [Axis(coll="events", field="deltaEtabb_min", bins=50, start=0, stop=5,
                  label="$\Delta \eta_{bb}$")]
        ),
        "deltaPhibb_min" : HistConf(
            [Axis(coll="events", field="deltaPhibb_min", bins=50, start=0, stop=5,
                  label="$\Delta \phi_{bb}$")]
        ),
        "mbb_closest" : HistConf(
            [Axis(coll="events", field="mbb_closest", bins=50, start=0, stop=500,
                    label="$m_{bb}(min \Delta R(bb))$ [GeV]")]
        ),
        "mbb_min" : HistConf(
            [Axis(coll="events", field="mbb_min", bins=50, start=0, stop=500,
                    label="$m_{bb}^{min}$ [GeV]")]
        ),
        "mbb_max" : HistConf(
            [Axis(coll="events", field="mbb_max", bins=50, start=0, stop=500,
                    label="$m_{bb}^{max}$ [GeV]")]
        ),
        "deltaRbb_avg" : HistConf(
            [Axis(coll="events", field="deltaRbb_avg", bins=50, start=0, stop=5,
                  label="$\Delta R_{bb}^{avg}$")]
        ),
        "spanet_tthbb" : HistConf(
            [Axis(coll="spanet_output", field="tthbb", bins=50, start=0, stop=1, label="tthbb SPANet score")],
        ),
        "spanet_tthbb_transformed" : HistConf(
            [Axis(coll="spanet_output", field="tthbb_transformed", bins=50, start=0, stop=1, label="tthbb SPANet transformed score")],
        ),
        "spanet_ttbb" : HistConf(
            [Axis(coll="spanet_output", field="ttbb", bins=50, start=0, stop=1, label="ttbb SPANet score")],
        ),
        "spanet_ttcc" : HistConf(
            [Axis(coll="spanet_output", field="ttcc", bins=50, start=0, stop=1, label="ttcc SPANet score")],
        ),
        "spanet_ttlf" : HistConf(
            [Axis(coll="spanet_output", field="ttlf", bins=50, start=0, stop=1, label="ttlf SPANet score")],
        )
    },
)

# Registering custom functions
import cloudpickle
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(workflow_spanet)
cloudpickle.register_pickle_by_value(workflow_control_regions)
cloudpickle.register_pickle_by_value(custom_cut_functions)
cloudpickle.register_pickle_by_value(custom_cuts)
cloudpickle.register_pickle_by_value(quantile_transformer)
