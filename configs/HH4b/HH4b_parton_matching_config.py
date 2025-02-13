# for spanet evaluation: pocket-coffea run --cfg HH4b_parton_matching_config.py -e dask@T3_CH_PSI --custom-run-options params/t3_run_options.yaml -o /work/mmalucch/out_test --executor-custom-setup onnx_executor.py
import os
import sys

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
)
# from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters import defaults

from workflow import HH4bbQuarkMatchingProcessor

from configs.HH4b_common.custom_cuts_common import hh4b_presel, hh4b_presel_tight, hh4b_4b_region, hh4b_2b_region, hh4b_signal_region, hh4b_control_region, signal_region_run2, control_region_run2
from configs.HH4b_common.configurator_options import get_variables_dict, get_columns_list, DEFAULT_COLUMNS


localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters

CLASSIFICATION = False
TIGHT_CUTS = False
RANDOM_PT = True
SAVE_CHUNK = False

print("CLASSIFICATION ", CLASSIFICATION)
print("TIGHT_CUTS ", TIGHT_CUTS)
print("RANDOM_PT ", RANDOM_PT)

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = "2022_postEE"
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    # f"{localdir}/params/plotting_style.yaml",
    update=True,
)

onnx_model_dict={
    "SPANET_MODEL": "params/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx",
    "VBF_GGF_DNN_MODEL":"",
    # "VBF_GGF_DNN_MODEL":"/t3home/rcereghetti/ML_pytorch/out/20241212_223142_SemitTightPtLearningRateConstant/models/model_28.onnx",
    "BKG_MORPHING_DNN_MODEL": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_morphing/average_model_from_keras.onnx",
    "SIG_BKG_DNN_MODEL": "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_SvsB/model_fold0.onnx",
}

print(onnx_model_dict)


workflow_options = {
        "parton_jet_min_dR": 0.4,
        "max_num_jets": 5,
        "which_bquark": "last",
        "classification": CLASSIFICATION, 
        "tight_cuts": TIGHT_CUTS,
        "fifth_jet": "pt",
        "random_pt": RANDOM_PT,
        "rand_type": 0.3
    }
workflow_options.update(
    onnx_model_dict
)
if SAVE_CHUNK:
    workflow_options["dump_columns_as_arrays_per_chunk"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/training_samples/GluGlutoHHto4B_spanet_loose_03_17"
    # workflow_options["dump_columns_as_arrays_per_chunk"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_btag_ordering"
    # workflow_options["dump_columns_as_arrays_per_chunk"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_no_btag"


variables_dict = get_variables_dict(CLASSIFICATION, RANDOM_PT, False)

column_dict = DEFAULT_COLUMNS
event_cols = []
if CLASSIFICATION:
    event_cols += ["best_pairing_probability", "second_best_pairing_probability", "Delta_pairing_probabilities"]
if RANDOM_PT:
    for key in column_dict.keys():
        if not "Matched" in key:
            column_dict[key] += ["pt_orig", "mass_orig"]
    column_dict["events"] = ["random_pt_weights"]

column_list = get_columns_list(column_dict, not SAVE_CHUNK)


cfg = Configurator(
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_skimmed",
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/signal_ggF_HH4b.json",
            f"{localdir}/datasets/DATA_JetMET.json",
            f"{localdir}/datasets/DATA_JetMET_skimmed.json",
            f"{localdir}/datasets/QCD.json",
            f"{localdir}/datasets/SPANet_classification.json",
            f"{localdir}/datasets/signal_ggF_HH4b_local.json",
            f"{localdir}/datasets/signal_VBF_HH4b_local.json",
        ],
        "filter": {
            "samples": (
                [
                    # "GluGlutoHHto4B",
                    # "QCD-4Jets",
                    # "DATA_JetMET_JMENano",
                    # "DATA_JetMET_JMENano_skimmed",
                    # "SPANet_classification",
                    # "SPANet_classification_data",
                    # "GluGlutoHHto4B_poisson",
                    # "GluGlutoHHto4B_private",
                    "GluGlutoHHto4B_spanet",
                ]
                if CLASSIFICATION
                # else ["DATA_JetMET_JMENano"]
                else ["GluGlutoHHto4B_spanet"]
                # else ["GluGlutoHHto4B"]
            ),
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=HH4bbQuarkMatchingProcessor,
    workflow_options=workflow_options,
    skim=[
        get_HLTsel(primaryDatasets=["JetMET"]),
    ],
    preselections=[
        hh4b_presel if TIGHT_CUTS is False else hh4b_presel_tight
    ],
    categories={
        "4b_region": [hh4b_4b_region],  
        "4b_signal_region": [hh4b_4b_region, hh4b_signal_region],  
        "4b_control_region": [hh4b_4b_region, hh4b_control_region],  
        "4b_signal_region_run2": [hh4b_4b_region, signal_region_run2],  
        "4b_control_region_run2": [hh4b_4b_region, control_region_run2],  
        "2b_region": [hh4b_2b_region],
        "2b_signal_region": [hh4b_2b_region, hh4b_signal_region],  
        "2b_control_region": [hh4b_2b_region, hh4b_control_region],  
        "2b_signal_region_run2": [hh4b_2b_region, signal_region_run2],  
        "2b_control_region_run2": [hh4b_2b_region, control_region_run2],  
    },
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {},
            },
            "bysample": {},
        }
    },
    variables=variables_dict,
    columns={
        "common": {
            "inclusive": column_list
        },
        "bysample": {},
    },
)
