import awkward as ak
from pocket_coffea.lib.cut_definition import Cut


def semileptonic(events, params, year, sample, **kwargs):

    MET = events[params["METbranch"][year]]

    has_one_electron = events.nElectronGood == 1
    has_one_muon = events.nMuonGood == 1

    mask = (
        (events.nLeptonGood == 1)
        &
        # Here we properly distinguish between leading muon and leading electron
        (
            (
                has_one_electron
                & (
                    ak.firsts(events.LeptonGood.pt)
                    > params["pt_leading_electron"][year]
                )
            )
            | (
                has_one_muon
                & (ak.firsts(events.LeptonGood.pt) > params["pt_leading_muon"][year])
            )
        )
        & (events.nJetGood >= params["njet"])
        & (events.nBJetGood >= params["nbjet"])
        & (MET.pt > params["met"])
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)
