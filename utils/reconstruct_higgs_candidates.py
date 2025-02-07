import awkward as ak
import numpy as np
import sys

from math import sqrt

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

def possible_higgs_reco(jets, idx_collection):
    """
    Currently `jets` is "JetsGoodHiggs", I just wanted to keep it modular.

    But basically this function just combines the 4 jets into every possible combination of Higgses.
    There are 6 different higgses that are combined to 3 different HH pairs.

    idx_collection is defining what the possible invariant combinations are.
    comb_idx = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]] This has to be the same always...
    """
    if len(jets) == 0:
        higgs_candidates_unflatten_order = ak.Array(
            [[[], []], [[], []], [[], []]]
        )
        return higgs_candidates_unflatten_order

    higgs = {}
    # First number indicates to which pairing option the higgs belongs
    # The second number defines, if it is the first or the second higgs
    for pairing in [0,1,2]:
        for higgs_pos in [0,1]:
            higgs[f"{pairing}{higgs_pos}"] = ak.unflatten(
                        jets[:, idx_collection[pairing][higgs_pos][0]]
                        + jets[:, idx_collection[pairing][higgs_pos][1]],
                        1,
                    )
    # Indices from comb_idx !! NOT CORRESPONDING TO b-jets !!
    higgs_pair_0 = ak.concatenate([higgs["00"], higgs["01"]], axis=1)
    higgs_pair_1 = ak.concatenate([higgs["10"], higgs["11"]], axis=1)
    higgs_pair_2 = ak.concatenate([higgs["20"], higgs["21"]], axis=1)

    higgs_candidates = ak.concatenate(
        [higgs_pair_0, higgs_pair_1, higgs_pair_2], axis=1
    )
    higgs_candidates_unflatten = ak.unflatten(higgs_candidates, 2, axis=1)

    # order the higgs candidates by pt
    higgs_candidates_unflatten_order_idx = ak.argsort(
        higgs_candidates_unflatten.pt, axis=2, ascending=False
    )
    higgs_candidates_unflatten_order = higgs_candidates_unflatten[
        higgs_candidates_unflatten_order_idx
    ]
    return higgs_candidates_unflatten_order

def distance_pt_func(higgs_pair, k):
    if len(higgs_pair[0, 0]) == 0:
        return np.array([])
    higgs1 = higgs_pair[:, :, 0]
    higgs2 = higgs_pair[:, :, 1]
    dist = abs(higgs1.mass - higgs2.mass * k) / sqrt(1 + k**2)
    max_pt = np.maximum(higgs1.pt,higgs2.pt)
    return dist, max_pt

#def pt_list(higgs_pair):
#    if len(higgs_pair[0,0]) == 0:
#        return np.array([])
#    higgs1 = higgs_pair[:,:,0]
#    higgs2 = higgs_pair[:,:,1]
#    # Find maximal pt for each higgs_pairing and just give it back as a nested list 
#    # (then we can do the same as with the distance term):
#    #max_pt = [[max(h1_comb.pt,h2_comb.pt) for h1_comb,h2_comb in zip(h1,h2)] for h1,h2 in zip(higgs1,higgs2)]
#    return max_pt


def run2_matching_algorithm(jet_collection):
    # implement the Run 2 pairing algorithm
    # TODO: extend to 5 jets cases (more comb idx)
    comb_idx = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]

    higgs_candidates_unflatten_order = possible_higgs_reco(
        jet_collection, comb_idx
    )
    distance, max_pt = distance_pt_func(
        higgs_candidates_unflatten_order,
        1.04,
    )

    # order by distance and find the index of the smallest distance
    dist_order_idx = ak.argsort(distance, axis=1, ascending=True)
    dist_order = ak.sort(distance, axis=1, ascending=True)

    delta_dhh = abs(dist_order[:, 0] - dist_order[:, 1])

    # Do the pt ordering for all events and fill later only the ones where the dhh is smaller than 30
    pt_order_idx = ak.argsort(max_pt, axis=1, ascending=False)
    # pt_order = ak.sort(max_pt_list, axis=1, ascending=False)  # Maybe useful for debugging
    
    # Find idxes where the delta_dhh is smaller than 30 and if so, fill in by pt ordering.
    min_idx = ak.where(delta_dhh > 30, dist_order_idx[:,0], pt_order_idx[:,0])
            
    # min_idx = dist_order_idx[:, 0]

    leadingHiggs = add_fields(higgs_candidates_unflatten_order[np.arange(len(jet_collection)), min_idx][:,0])
    subleadingHiggs = add_fields(higgs_candidates_unflatten_order[np.arange(len(jet_collection)), min_idx][:,1])

    return (
        delta_dhh,
        leadingHiggs,
        subleadingHiggs,
        jet_collection[
            ak.Array([np.reshape(comb_idx[best], 4) for best in min_idx])
        ],
    )
