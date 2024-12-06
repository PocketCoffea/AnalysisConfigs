import awkward as ak
import numpy as np
import sys

sys.path.append("../")
from utils.basic_functions import add_fields


def reconstruct_higgs_from_provenance(matched_jets_higgs):

    jet_higgs1 = matched_jets_higgs[matched_jets_higgs.provenance == 1]
    jet_higgs2 = matched_jets_higgs[matched_jets_higgs.provenance == 2]

    jet_higgs1 = jet_higgs1[ak.argsort(jet_higgs1.pt, axis=1, ascending=False)]
    jet_higgs2 = jet_higgs2[ak.argsort(jet_higgs2.pt, axis=1, ascending=False)]

    higgs_lead = add_fields(jet_higgs1[:, 0] + jet_higgs1[:, 1])
    higgs_sub = add_fields(jet_higgs2[:, 0] + jet_higgs2[:, 1])

    jets_ordered = ak.with_name(
        ak.concatenate([jet_higgs1[:, :2], jet_higgs2[:, :2]], axis=1),
        name="PtEtaPhiMCandidate",
    )

    return higgs_lead, higgs_sub, jets_ordered


def reconstruct_higgs_from_idx(jet_collection, idx_collection):
    # NOTE: idx_collection is referred to the index of the jets in the JetGood collection
    # and is not the index of the jets in the Jet
    higgs_1 = ak.unflatten(
        jet_collection[np.arange(len(idx_collection)), idx_collection[:, 0, 0]]
        + jet_collection[np.arange(len(idx_collection)), idx_collection[:, 0, 1]],
        1,
    )
    higgs_2 = ak.unflatten(
        jet_collection[np.arange(len(idx_collection)), idx_collection[:, 1, 0]]
        + jet_collection[np.arange(len(idx_collection)), idx_collection[:, 1, 1]],
        1,
    )


    higgs_leading_index = ak.where(higgs_1.pt > higgs_2.pt, 0, 1)

    higgs_lead = ak.where(higgs_leading_index == 0, higgs_1, higgs_2)
    higgs_sub = ak.where(higgs_leading_index == 0, higgs_2, higgs_1)

    higgs_leading_index_expanded = higgs_leading_index[:, np.newaxis] * np.ones((2, 2))
    # order idx according to the higgs candidate
    idx_ordered = ak.where(
        higgs_leading_index_expanded == 0, idx_collection, idx_collection[:, ::-1]
    )

    higgs1_jet1 = ak.unflatten(
        ak.where(
            jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 0, 0]].pt
            > jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 0, 1]].pt,
            jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 0, 0]],
            jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 0, 1]],
        ),1
    )
    higgs1_jet2 = ak.unflatten(
        ak.where(
            jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 0, 0]].pt
            > jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 0, 1]].pt,
            jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 0, 1]],
            jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 0, 0]],
        ),1
    )
    higgs2_jet1 = ak.unflatten(
        ak.where(
            jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 1, 0]].pt
            > jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 1, 1]].pt,
            jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 1, 0]],
            jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 1, 1]],
        ),1
    )
    higgs2_jet2 = ak.unflatten(
        ak.where(
            jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 1, 0]].pt
            > jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 1, 1]].pt,
            jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 1, 1]],
            jet_collection[np.arange(len(idx_collection)), idx_ordered[:, 1, 0]],
        ),1
    )

    jets_ordered = ak.with_name(
        ak.concatenate([higgs1_jet1, higgs1_jet2, higgs2_jet1, higgs2_jet2], axis=1),
        name="PtEtaPhiMCandidate",
    )

    higgs_lead = add_fields(ak.flatten(higgs_lead))
    higgs_sub = add_fields(ak.flatten(higgs_sub))

    return higgs_lead, higgs_sub, jets_ordered
