from collections.abc import Iterable
import awkward as ak
import custom_cut_functions as cuts_f
from pocket_coffea.lib.cut_definition import Cut

semileptonic_presel = Cut(
    name="semileptonic",
    params={
        "njet": 4,
        "nbjet": 3,
        "pt_leading_electron": {
            '2016_PreVFP': 29,
            '2016_PostVFP': 29,
            '2017': 30,
            '2018': 30,
        },
        "pt_leading_muon": {
            '2016_PreVFP': 26,
            '2016_PostVFP': 26,
            '2017': 29,
            '2018': 26,
        },
        "met": 20,
    },
    function=cuts_f.semileptonic,
)

semilep_lhe = Cut(
    name="semilep_lhe",
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

# Selection for ttbar background categorization
def get_genTtbarId_100_eq(genTtbarId, name=None):
    if name == None:
        if type(genTtbarId) == int:
            name = f"genTtbarId_100_eq_{genTtbarId}"
        if isinstance(genTtbarId, Iterable):
            name = f"genTtbarId_100_eq_" + "_".join([str(s) for s in genTtbarId])
    return Cut(name=name, params={"genTtbarId" : genTtbarId}, function=cuts_f.eq_genTtbarId_100)
