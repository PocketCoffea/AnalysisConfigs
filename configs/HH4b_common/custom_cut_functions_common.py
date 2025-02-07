from collections.abc import Iterable
import awkward as ak
import numpy as np


def hh4b_presel_cuts(events, params, **kwargs):
    at_least_four_jets = events.nJetGood >= params["njet"]
    no_electron = events.nElectronGood == 0
    no_muon = events.nMuonGood == 0

    mask_4jet_nolep = at_least_four_jets & no_electron & no_muon

    # convert false to None
    mask_4jet_nolep_none = ak.mask(mask_4jet_nolep, mask_4jet_nolep)
    jets_btag_order = (
        events[mask_4jet_nolep_none].JetGood
        if not params["tight_cuts"]
        else events[mask_4jet_nolep_none].JetGoodHiggs
    )

    jets_pt_order = jets_btag_order[
        ak.argsort(jets_btag_order.pt, axis=1, ascending=False)
    ]

    mask_pt_none = (
        (jets_pt_order.pt[:, 0] > params["pt_jet0"])
        & (jets_pt_order.pt[:, 1] > params["pt_jet1"])
        & (jets_pt_order.pt[:, 2] > params["pt_jet2"])
        & (jets_pt_order.pt[:, 3] > params["pt_jet3"])
    )
    # convert none to false
    mask_pt = ak.where(ak.is_none(mask_pt_none), False, mask_pt_none)

    mask_btag = (
        (jets_btag_order.btagPNetB[:, 0] + jets_btag_order.btagPNetB[:, 1]) / 2
        > params["mean_pnet_jet"]
        # & (jets_btag_order.btagPNetB[:, 2] > params["third_pnet_jet"])
        # & (jets_btag_order.btagPNetB[:, 3] > params["fourth_pnet_jet"])
    )

    mask_btag = ak.where(ak.is_none(mask_btag), False, mask_btag)

    mask = mask_pt & mask_btag

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_2b_cuts(events, params, **kwargs):
    jets_btag_order = events.JetGoodHiggs

    mask = (jets_btag_order.btagPNetB[:, 2] < params["third_pnet_jet"]) & (
        jets_btag_order.btagPNetB[:, 3] < params["fourth_pnet_jet"]
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_4b_cuts(events, params, **kwargs):
    jets_btag_order = events.JetGoodHiggs

    mask = (jets_btag_order.btagPNetB[:, 2] > params["third_pnet_jet"]) & (
        jets_btag_order.btagPNetB[:, 3] > params["fourth_pnet_jet"]
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)


def hh4b_Rhh_cuts(events, params, **kwargs):

    mask = (events.Rhh >= params["radius_min"]) & (events.Rhh < params["radius_max"])

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

