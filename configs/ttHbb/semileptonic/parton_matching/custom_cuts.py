# Per-event cuts applied to each event
import awkward as ak
import custom_cut_functions as cuts_f
from pocket_coffea.lib.cut_definition import Cut



semileptonic_presel = Cut(
    name="semileptonic",
    params={
        "METbranch": {
            '2016_PreVFP': "MET",
            '2016_PostVFP': "MET",
            '2017': "MET",
            '2018': "MET",
        },
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

# Same preselection as the standard semileptonic but without
# requirements on the number of btagged jets
# --> used for btagSF normalization calibration
semileptonic_presel_nobtag = Cut(
    name="semileptonic_nobtag",
    params={
        "METbranch": {
            '2016_PreVFP': "MET",
            '2016_PostVFP': "MET",
            '2017': "MET",
            '2018': "MET",
        },
        "njet": 4,
        "nbjet": 0,
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

