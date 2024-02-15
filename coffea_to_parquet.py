import os
import argparse
from collections import defaultdict

import numpy as np
import awkward as ak

import vector

vector.register_numba()
vector.register_awkward()

from coffea.util import load
from coffea.processor.accumulator import column_accumulator
from coffea.processor import accumulate

# Read arguments from command line: input file and output directory. Description: script to convert ntuples from coffea file to parquet file.
parser = argparse.ArgumentParser(
    description="Convert awkward ntuples in coffea files to parquet files."
)
parser.add_argument("-i", "--input", type=str, required=True, help="Input coffea file")
parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
parser.add_argument(
    "-c",
    "--cat",
    type=str,
    default="baseline",
    required=False,
    help="Event category",
)

args = parser.parse_args()

## Loading the exported dataset
# We open the .coffea file and read the output accumulator. The ntuples for the training are saved under the key `columns`.

if not os.path.exists(args.input):
    raise ValueError(f"Input file {args.input} does not exist.")
if not os.path.exists(args.output):
    os.makedirs(args.output)

df = load(args.input)

if not args.cat in df["cutflow"].keys():
    raise ValueError(f"Event category `{args.cat}` not found in the input file.")

# Dictionary of features to be used for the training
# The dictionary has two levels: the first level is common to all the samples, the second level is specific for a given sample.
# For each of these levels, the dictionary contains the name of the collection (e.g. `JetGood`) and the features to be used for the training (e.g. `pt`, `eta`, `phi`, `mass`, `btag`).
# For each feature, the dictionary contains the name of the feature in the coffea file (e.g. `provenance`) and the name of the feature in the parquet file (e.g. `prov`).

features = {
    "common": {
        # "bQuark": {
        #     "pt": "pt",
        #     "eta": "eta",
        #     "phi": "phi",
        #     # "mass" : "mass",
        #     "pdgId": "pdgId",
        #     "prov": "provenance",
        # },
        # "bQuarkHiggsMatched": {
        #     "pt": "pt",
        #     "eta": "eta",
        #     "phi": "phi",
        #     # "mass" : "mass",
        #     "pdgId": "pdgId",
        #     "prov": "provenance",
        # },
        "JetGood": {
            "pt": "pt",
            "eta": "eta",
            "phi": "phi",
            "btag": "btagPNetB",
            "ptPnetRegNeutrino": "ptPnetRegNeutrino",
        },
        "JetGoodHiggs": {
            "pt": "pt",
            "eta": "eta",
            "phi": "phi",
            "btag": "btagPNetB",
            "ptPnetRegNeutrino": "ptPnetRegNeutrino",
        },
        "JetGoodHiggsMatched": {
            "pt": "pt",
            "eta": "eta",
            "phi": "phi",
            "btag": "btagPNetB",
            "ptPnetRegNeutrino": "ptPnetRegNeutrino",
            "prov": "provenance",
            # "pdgId" : "pdgId",
            # "hadronFlavour" : "hadronFlavour"
        },
        "JetGoodMatched": {
            "pt": "pt",
            "eta": "eta",
            "phi": "phi",
            "btag": "btagPNetB",
            "ptPnetRegNeutrino": "ptPnetRegNeutrino",
            "prov": "provenance",
            # "pdgId" : "pdgId",
            # "hadronFlavour" : "hadronFlavour"
        },
    },
    "by_sample": {},
}

# Dictionary of features to pad with a default value
features_pad = {
    "common": {
        # "JetGood" : {
        #     "m" : 0
        # },
        # "JetGoodHiggs" : {
        #     "m" : 0
        # },
        # "JetGoodHiggsMatched" : {
        #     "m" : 0
        # },
    },
    "by_sample": {},
}

awkward_collections = list(features["common"].keys())
matched_collections_dict = {
    # "bQuark": "bQuarkHiggsMatched",
    "JetGoodHiggs": "JetGoodHiggsMatched",
    "JetGood": "JetGoodMatched",
}

samples = df["columns"].keys()
print("Samples: ", samples)

for sample in samples:

    # Compose the features dictionary with common features and sample-specific features
    features_dict = features["common"].copy()
    if sample in features["by_sample"].keys():
        features_dict.update(features["by_sample"][sample])

    # Compose the dictionary of features to pad
    features_pad_dict = features_pad["common"].copy()
    if sample in features_pad["by_sample"].keys():
        features_pad_dict.update(features_pad["by_sample"][sample])

    # Create a default dictionary of dictionaries to store the arrays
    array_dict = {k: defaultdict(dict) for k in features_dict.keys()}
    datasets = df["columns"][sample].keys()
    print("Datasets: ", datasets)

    ## Normalize the genweights
    # Since the array `weight` is filled on the fly with the weight associated with the event, it does not take into account the overall scaling by the sum of genweights (`sum_genweights`).
    # In order to correct for this, we have to scale by hand the `weight` array dividing by the sum of genweights.
    for dataset in datasets:
        weight = df["columns"][sample][dataset][args.cat]["weight"].value
        weight_new = column_accumulator(weight / df["sum_genweights"][dataset])
        df["columns"][sample][dataset][args.cat]["weight"] = weight_new

    ## Accumulate ntuples from different data-taking eras
    # In order to enlarge our training sample, we merge ntuples coming from different data-taking eras.
    cs = accumulate([df["columns"][sample][dataset][args.cat] for dataset in datasets])

    ## Build the Momentum4D arrays for the jets, partons, leptons, met and higgs
    # In order to get the numpy array from the column_accumulator, we have to access the `value` attribute.
    for collection, variables in features_dict.items():
        for key_feature, key_coffea in variables.items():
            # if (collection == "JetGoodHiggsMatched") & (key_coffea == "provenance"):
            #     array_dict[collection][key_feature] = cs[f"bQuarkHiggsMatched_{key_coffea}"].value
            # else:
            array_dict[collection][key_feature] = cs[f"{collection}_{key_coffea}"].value

        # Add padded features to the array, according to the features dictionary
        if collection in features_pad_dict.keys():
            for key_feature, value in features_pad_dict[collection].items():
                array_dict[collection][key_feature] = value * np.ones_like(
                    cs[f"{collection}_pt"].value
                )

    # The awkward arrays are zipped together to form the Momentum4D arrays.
    # If the collection is not a Momentum4D array, it is zipped as it is,
    # otherwise the Momentum4D arrays are zipped together and unflattened depending on the number of objects in the collection.
    zipped_dict = {}
    for collection in array_dict.keys():
        if collection in awkward_collections:
            zipped_dict[collection] = ak.unflatten(
                ak.zip(array_dict[collection], with_name="Momentum4D"),
                cs[f"{collection}_N"].value,
            )
        else:
            zipped_dict[collection] = ak.zip(
                array_dict[collection], with_name="Momentum4D"
            )
        print(f"Collection: {collection}")
        print("Fields: ", zipped_dict[collection].fields)

    for collection in zipped_dict.keys():
        # Pad the matched collections with None if there is no matching
        if collection in matched_collections_dict.keys():
            matched_collection = matched_collections_dict[collection]
            masked_arrays = ak.mask(
                zipped_dict[matched_collection],
                zipped_dict[matched_collection].pt == -999,
                None,
            )
            print("masked_arrays: ", masked_arrays)
            zipped_dict[matched_collection] = masked_arrays
            # Add the matched flag and the provenance to the matched jets
            if collection == "JetGoodHiggs"or collection ==  "JetGood":
                print(
                    "Adding the matched flag and the provenance to the matched jets..."
                )
                is_matched = ~ak.is_none(masked_arrays, axis=1)
                print("is_matched: ", is_matched)
                print("zipped: ", zipped_dict[collection].pt, zipped_dict[collection])
                zipped_dict[collection] = ak.with_field(
                    zipped_dict[collection], is_matched, "matched"
                )
                zipped_dict[collection] = ak.with_field(
                    zipped_dict[collection],
                    ak.fill_none(masked_arrays.prov, -1),
                    "prov",
                )

    # The Momentum4D arrays are zipped together to form the final dictionary of arrays.
    print("Zipping the collections into a single dictionary...")
    df_out = ak.zip(zipped_dict, depth_limit=1)
    filename = os.path.join(args.output, f"{sample}.parquet")
    print(f"Saving the output dataset to file: {os.path.abspath(filename)}")
    ak.to_parquet(df_out, filename)
