import numpy as np
import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

def trigger_mask(events, params, **kwargs):
    mask = np.zeros(len(events), dtype='bool')
    for trigger in params["triggers"]:
        mask = mask | events.HLT[trigger]
    assert (params["category"] in ["pass", "fail"]), "The allowed categories for the trigger selection are 'pass' and 'fail'"
    if params["category"] == "fail":
        mask = ~mask
    return mask

def trigger_mask_2017(events, params, **kwargs):
    mask = np.zeros(len(events), dtype='bool')
    for trigger in params["triggers"]:
        if not trigger in events.HLT.fields: continue
        if trigger != 'Ele32_WPTight_Gsf_L1DoubleEG':
            mask = mask | events.HLT[trigger]
        elif ((trigger == 'Ele32_WPTight_Gsf_L1DoubleEG') & ('Ele32_WPTight' not in events.HLT.fields)):
            flag = ak.sum( (events.TrigObj.id == 11) & ((events.TrigObj.filterBits & 1024) == 1024), axis=1 ) > 0
            mask = mask | ( events.HLT[trigger] & flag )
    assert (params["category"] in ["pass", "fail"]), "The allowed categories for the trigger selection are 'pass' and 'fail'"
    if params["category"] == "fail":
        mask = ~mask
    return mask

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
