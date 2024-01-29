from collections.abc import Iterable
import awkward as ak

def semileptonic(events, params, year, sample, **kwargs):

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
        & (events.MET.pt > params["met"])
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

def eq_genTtbarId_100(events, params, year, sample, **kwargs):
    """
    This function returns a mask for events where genTtbarId % 100 == params["genTtbarId"]
    or the logic OR of masks in the case in which params["genTtbarId"] is an iterable.
    The dictionary for genTtbarId % 100 is the following
    (taken from https://twiki.cern.ch/twiki/bin/view/CMSPublic/GenHFHadronMatcher, visited on 23.11.2023):
    0  : "tt+LF",
    41 : "tt+c",
    42 : "tt+2c",
    43 : "tt+cc",
    44 : "tt+c2c",
    45 : "tt+2c2c",
    46 : "tt+C",
    51 : "tt+b",
    52 : "tt+2b",
    53 : "tt+bb",
    54 : "tt+b2b",
    55 : "tt+2b2b",
    56 : "tt+B",
    """
    allowed_ids = [0, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56]
    if type(params["genTtbarId"]) == int:
        if params["genTtbarId"] in allowed_ids:
            return events.genTtbarId % 100 == params["genTtbarId"]
        else:
            raise Exception(f"The cut on genTtbarId % 100 must be an integer between 0 and 56.\nPossible choices:{allowed_ids}")
    elif isinstance(params["genTtbarId"], Iterable):
        mask = ak.zeros_like(events.event, dtype=bool)
        for _id in params["genTtbarId"]:
            if _id in allowed_ids:
                mask = mask | (events.genTtbarId % 100 == _id)
            else:
                raise Exception(f"The cut on genTtbarId % 100 must be an integer between 0 and 56.\nPossible choices:{allowed_ids}")
        return mask
    else:
        raise Exception(f'params["genTtbarId"] must be an integer or an iterable of integers between 0 and 56.\nPossible choices:{allowed_ids}')
