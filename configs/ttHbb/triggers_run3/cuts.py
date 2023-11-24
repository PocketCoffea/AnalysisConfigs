from pocket_coffea.lib.cut_definition import Cut
import awkward as ak

semilep_lhe = Cut(
    name="semilep_lhe",
    params={},
    function=lambda events, params, **kwargs: ak.sum( (abs(events.LHEPart.pdgId) >=11)&(abs(events.LHEPart.pdgId) <17), axis=1)==2
)

dilep_lhe = Cut(
    name="dilep_lhe",
    params={},
    function=lambda events, params, **kwargs: ak.sum( (abs(events.LHEPart.pdgId) >=11)&(abs(events.LHEPart.pdgId) <17), axis=1)==4
)

had_lhe = Cut(
    name="had_lhe",
    params={},
    function=lambda events, params, **kwargs:  ak.sum( (abs(events.LHEPart.pdgId) >=11)&(abs(events.LHEPart.pdgId) <17), axis=1)==0
)

notau = Cut(
    name="notau",
    params={},
    function=lambda events, params, **kwargs: ak.sum(abs(events.LHEPart.pdgId) == 15, axis=1)==0
)


