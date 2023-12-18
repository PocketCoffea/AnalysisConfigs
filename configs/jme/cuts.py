from pocket_coffea.lib.cut_definition import Cut
import awkward as ak

def ptbin(events, params, **kwargs):
    # Mask to select events in a MatchedJets pt bin
    if params["pt_high"] == "Inf":
        mask = events.MatchedJets.pt > params["pt_low"]
    elif type(params["pt_high"]) != str:
        mask = (events.MatchedJets.pt > params["pt_low"]) & (
            events.MatchedJets.pt < params["pt_high"]
        )
    else:
        raise NotImplementedError

    assert not ak.any(ak.is_none(mask, axis=1)), f"None in ptbin"#\n{events.nJetGood[ak.is_none(mask, axis=1)]}"

    return mask


def get_ptbin(pt_low, pt_high, name=None):
    if name == None:
        name = f"pt{pt_low}-{pt_high}"
    return Cut(
        name=name,
        params={"pt_low": pt_low, "pt_high": pt_high},
        function=ptbin,
        collection="MatchedJets",
    )


# do th same for eta
def etabin(events, params, **kwargs):
    # Mask to select events in a MatchedJets eta bin
    mask = (abs(events.MatchedJets.eta) > params["eta_low"]) & (
        abs(events.MatchedJets.eta) < params["eta_high"]
    )
    # substitute none with false in mask
    # mask = ak.where(ak.is_none(mask, axis=1), False, mask)
    mask[~ak.is_none(mask, axis=1)]



    assert not ak.any(ak.is_none(mask, axis=1)), f"None in etabin"#\n{events.nJetGood[ak.is_none(mask, axis=1)]}"

    return mask


def get_etabin(eta_low, eta_high, name=None):
    if name == None:
        name = f"eta{eta_low}-{eta_high}"
    return Cut(
        name=name,
        params={"eta_low": eta_low, "eta_high": eta_high},
        function=etabin,
        collection="MatchedJets",
    )
