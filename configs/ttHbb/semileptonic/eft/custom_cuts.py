# Per-event cuts applied to each event
import awkward as ak
from pocket_coffea.lib.cut_definition import Cut



semilep_lhe_notau = Cut(
    name="semilep_lhe_notau",
    params={},
    function=lambda events, params, **kwargs: (
        ak.sum(
            (abs(events.LHEPart.pdgId) >= 11) & (abs(events.LHEPart.pdgId) < 15), axis=1
        )
        == 2
    )
    & (
        ak.sum(
            (abs(events.LHEPart.pdgId) >= 15) & (abs(events.LHEPart.pdgId) < 19), axis=1
        )
        == 0
    ),
)

semilep_lhe = Cut(
    name="semilep_lhe",
    params={},
    function=lambda events, params, **kwargs: (
        ak.sum(
            (abs(events.LHEPart.pdgId) >= 11) & (abs(events.LHEPart.pdgId) < 19), axis=1
        )
        == 2
    )
)
