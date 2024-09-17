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
