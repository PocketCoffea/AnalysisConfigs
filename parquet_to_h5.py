import awkward as ak
import numba
import numpy as np
import pandas as pd
import awkward as ak
import os
import h5py
import vector
import argparse
from multiprocessing import Pool
import functools

vector.register_numba()
vector.register_awkward()


parser = argparse.ArgumentParser(
    description="Convert awkward ntuples in coffea files to parquet files."
)
parser.add_argument("-i", "--input", type=str, required=True, help="Input coffea file")
parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
parser.add_argument(
    "-f",
    "--frac-train",
    type=float,
    default=0.8,
    help="Fraction of events to use for training",
)


args = parser.parse_args()

filename = f"{args.input}"
main_dir = args.output
df = ak.from_parquet(filename)


def create_groups(file):
    file.create_group("TARGETS/h1")  # higgs 1 -> b1 b2
    file.create_group("TARGETS/h2")  # higgs 2 -> b3 b4
    file.create_group("INPUTS")
    # file.create_group("INPUTS/Source")
    # file.create_group("INPUTS/ht")
    return file


def create_targets(file, particle, jets, filename):
    multiindex = ak.zip([ak.local_index(jets, i) for i in range(jets.ndim)])

    higgs_targets = {1: ["b1", "b2"], 2: ["b3", "b4"]}

    for j in [1, 2]:
        if particle == f"h{j}":
            mask = jets.prov == j  # H->b1b2
            multiindex2 = multiindex[mask]
            print(filename, particle, multiindex2)

            b1_array = []
            b2_array = []

            for index, i in enumerate(multiindex2):
                if len(i) == 0:
                    b1_array.append(-1)
                    b2_array.append(-1)
                elif len(i) == 1:
                    b1_array.append(i[0].tolist()[1])
                    b2_array.append(-1)
                elif len(i) == 2:
                    b1_array.append(i[0].tolist()[1])
                    b2_array.append(i[1].tolist()[1])

            file.create_dataset(
                f"TARGETS/h{j}/{higgs_targets[j][0]}",
                np.shape(b1_array),
                dtype="int64",
                data=b1_array,
            )
            file.create_dataset(
                f"TARGETS/h{j}/{higgs_targets[j][1]}",
                np.shape(b2_array),
                dtype="int64",
                data=b2_array,
            )


def create_inputs(file, jets):
    pt_array = ak.to_numpy(ak.fill_none(ak.pad_none(jets.pt, 16, clip=True), 0))
    mask = ~(pt_array == 0)
    mask_ds = file.create_dataset(
        "INPUTS/Jet/MASK", np.shape(mask), dtype="bool", data=mask
    )
    pt_ds = file.create_dataset(
        "INPUTS/Jet/pt", np.shape(pt_array), dtype="float32", data=pt_array
    )

    ptPnetRegNeutrino_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.ptPnetRegNeutrino, 16, clip=True), 0)
    )
    ptPnetRegNeutrino_ds = file.create_dataset(
        "INPUTS/Jet/ptPnetRegNeutrino",
        np.shape(ptPnetRegNeutrino_array),
        dtype="float32",
        data=ptPnetRegNeutrino_array,
    )

    phi_array = ak.to_numpy(ak.fill_none(ak.pad_none(jets.phi, 16, clip=True), 0))
    phi_ds = file.create_dataset(
        "INPUTS/Jet/phi", np.shape(phi_array), dtype="float32", data=phi_array
    )

    eta_array = ak.to_numpy(ak.fill_none(ak.pad_none(jets.eta, 16, clip=True), 0))
    eta_ds = file.create_dataset(
        "INPUTS/Jet/eta", np.shape(eta_array), dtype="float32", data=eta_array
    )

    btag = ak.to_numpy(ak.fill_none(ak.pad_none(jets.btag, 16, clip=True), 0))
    btag_ds = file.create_dataset(
        "INPUTS/Jet/btag", np.shape(btag), dtype="float32", data=btag
    )

    # # Fill ht
    # pt_array = ak.to_numpy(ak.fill_none(ak.pad_none(jets.pt, 15, clip=True), 0))
    # ht_array = np.sum(pt_array, axis=1)
    # ht_ds = file.create_dataset(
    #     "INPUTS/ht/ht", np.shape(ht_array), dtype="float32", data=ht_array
    # )

    # # Fill ht
    # ptPnetRegNeutrino_array = ak.to_numpy(
    #     ak.fill_none(ak.pad_none(jets.ptPnetRegNeutrino, 15, clip=True), 0)
    # )
    # htPNetRegNeutrino_array = np.sum(ptPnetRegNeutrino_array, axis=1)
    # htPNetRegNeutrino_ds = file.create_dataset(
    #     "INPUTS/ht/htPNetRegNeutrino",
    #     np.shape(htPNetRegNeutrino_array),
    #     dtype="float32",
    #     data=htPNetRegNeutrino_array,
    # )


file_dict = {
    0: "output_JetGood_train.h5",
    1: "output_JetGood_test.h5",
    2: "output_JetGoodHiggs_train.h5",
    3: "output_JetGoodHiggs_test.h5",
}


def add_info_to_file(input_to_file):
    k, jets = input_to_file
    print(f"Adding info to file {file_dict[k]}")
    file_out = h5py.File(f"{main_dir}/{file_dict[k]}", "w")
    file_out = create_groups(file_out)
    create_inputs(file_out, jets)
    create_targets(file_out, "h1", jets, file_dict[k])
    create_targets(file_out, "h2", jets, file_dict[k])
    file_out.close()


# create the test and train datasets
# and create differnt datasets with jetGood and jetGoodHiggs

jets_good = df.JetGood
jets_good_higgs = df.JetGoodHiggs

jets_list = []
for i, jets_all in enumerate([jets_good, jets_good_higgs]):
    print(f"Creating dataset for {'JetGood' if i == 0 else 'JetGoodHiggs'}")
    n_events = len(jets_all)
    print(f"Number of events: {n_events}")
    idx_train_max = int(np.ceil(n_events * args.frac_train))
    print(f"Number of events for training: {idx_train_max}")
    print(f"Number of events for testing: {n_events - idx_train_max}")
    jets_train = jets_all[:idx_train_max]
    jets_test = jets_all[idx_train_max:]

    for j, jets in enumerate([jets_train, jets_test]):
        jets_list.append(jets)
        # if j == 0:
        #     print("Creating training dataset")
        #     file_out = h5py.File(
        #         f"{main_dir}/output_{'JetGood' if i == 0 else 'JetGoodHiggs'}_train.h5",
        #         "w",
        #     )
        # else:
        #     print("Creating testing dataset")
        #     file_out = h5py.File(
        #         f"{main_dir}/output_{'JetGood' if i == 0 else 'JetGoodHiggs'}_test.h5",
        #         "w",
        #     )
        # file_out = create_groups(file_out)
        # create_inputs(file_out, jets)
        # create_targets(file_out, "h1", jets)
        # create_targets(file_out, "h2", jets)

        # file_out.close()

with Pool(4) as p:
    p.map(add_info_to_file, enumerate(jets_list))
