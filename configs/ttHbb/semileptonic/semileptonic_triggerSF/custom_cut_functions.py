import numpy as np
import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

def ht_above(events, params, **kwargs):
    mask = events["JetGood_Ht"] > params["minht"]
    return mask

def ht_below(events, params, **kwargs):
    mask = (events["JetGood_Ht"] >= 0) & (events["JetGood_Ht"] < params["maxht"])
    return mask

def semileptonic_triggerSF(events, params, year, sample, **kwargs):

    has_one_electron = events.nElectronGood == 1
    has_one_muon = events.nMuonGood == 1

    mask = (
        (events.nLeptonGood == 2)
        &
        # Here we properly distinguish between leading muon and leading electron
        (
            (
                has_one_electron
                & (
                    ak.firsts(events.ElectronGood.pt)
                    > params["pt_leading_electron"][year]
                )
            )
            & (
                has_one_muon
                & (ak.firsts(events.MuonGood.pt) > params["pt_leading_muon"][year])
            )
        )
        & (events.nJetGood >= params["njet"])
    )
    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)
