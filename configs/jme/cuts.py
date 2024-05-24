from pocket_coffea.lib.cut_definition import Cut
import awkward as ak
import os


def ptbin(events, params, **kwargs):
    # Mask to select events in a MatchedJets pt bin
    if params["pt_high"] == "Inf":
        mask = events.MatchedJets.pt > params["pt_low"]
    elif type(params["pt_high"]) != str:
        mask = (events.MatchedJets.JetPtRaw > params["pt_low"]) & (  # HERE
            events.MatchedJets.JetPtRaw < params["pt_high"]
        )
    else:
        raise NotImplementedError

    # mask = ak.where(ak.is_none(mask, axis=1), False, mask)
    # mask=mask[~ak.is_none(mask, axis=1)]

    assert not ak.any(
        ak.is_none(mask, axis=1)
    ), f"None in ptbin"  # \n{events.nJetGood[ak.is_none(mask, axis=1)]}"

    return mask


def get_ptbin(pt_low, pt_high, name=None):
    if name == None:
        name = f"pt{pt_low}to{pt_high}"
    return Cut(
        name=name,
        params={"pt_low": pt_low, "pt_high": pt_high},
        function=ptbin,
        collection="MatchedJets",
    )


def etabin(events, params, **kwargs):
    # Mask to select events in a MatchedJets eta bin
    mask = (
        (events.MatchedJets.eta > params["eta_low"])
        & (events.MatchedJets.eta < params["eta_high"])
    )

    # mask = ak.where(ak.is_none(mask, axis=1), False, mask)
    # mask=mask[~ak.is_none(mask, axis=1)]

    assert not ak.any(
        ak.is_none(mask, axis=1)
    ), f"None in etabin"  # \n{events.nJetGood[ak.is_none(mask, axis=1)]}"

    return mask

def reco_etabin(events, params, **kwargs):
    # Mask to select events in a MatchedJets eta bin
    mask = (
        (events.MatchedJetsNeutrino.RecoEta > params["eta_low"])
        & (events.MatchedJetsNeutrino.RecoEta < params["eta_high"])
    )

    # mask = ak.where(ak.is_none(mask, axis=1), False, mask)
    # mask=mask[~ak.is_none(mask, axis=1)]

    assert not ak.any(
        ak.is_none(mask, axis=1)
    ), f"None in etabin"  # \n{events.nJetGood[ak.is_none(mask, axis=1)]}"

    return mask


def etabin_neutrino(events, params, **kwargs):
    # Mask to select events in a MatchedJets eta bin
    mask = (events.MatchedJetsNeutrino.eta > params["eta_low"]) & (
        events.MatchedJetsNeutrino.eta < params["eta_high"]
    )

    # mask = ak.where(ak.is_none(mask, axis=1), False, mask)
    # mask=mask[~ak.is_none(mask, axis=1)]

    assert not ak.any(
        ak.is_none(mask, axis=1)
    ), f"None in etabin"  # \n{events.nJetGood[ak.is_none(mask, axis=1)]}"

    return mask


def get_etabin(eta_low, eta_high, name=None):
    if name == None:
        name = f"MatchedJets_eta{eta_low}to{eta_high}"
    return Cut(
        name=name,
        params={"eta_low": eta_low, "eta_high": eta_high},
        function=etabin,
        collection=(
            "MatchedJets"
        ),
    )


def get_etabin_neutrino(eta_low, eta_high, name=None):
    if name == None:
        name = f"MatchedJetsNeutrino_eta{eta_low}to{eta_high}"
    return Cut(
        name=name,
        params={"eta_low": eta_low, "eta_high": eta_high},
        function=etabin_neutrino,
        collection="MatchedJetsNeutrino",
    )

def get_reco_etabin(eta_low, eta_high, name=None):
    if name == None:
        name = f"MatchedJetsNeutrino_reco_eta{eta_low}to{eta_high}"
    return Cut(
        name=name,
        params={"eta_low": eta_low, "eta_high": eta_high},
        function=reco_etabin,
        collection=(
            "MatchedJetsNeutrino"
        ),
    )