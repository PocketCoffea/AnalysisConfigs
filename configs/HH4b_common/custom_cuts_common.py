from . import custom_cut_functions_common as cuts_f
from pocket_coffea.lib.cut_definition import Cut


hh4b_presel = Cut(
    name="hh4b_presel",
    params={
        "njet": 4,
        "pt_jet0": 80,
        "pt_jet1": 60,
        "pt_jet2": 45,
        "pt_jet3": 35,
        "mean_pnet_jet": 0.65,
        "tight_cuts": False,
    },
    function=cuts_f.hh4b_presel_cuts,
)

hh4b_presel_tight = Cut(
    name="hh4b_presel_tight",
    params={
        "njet": 4,
        "pt_jet0": 80,
        "pt_jet1": 60,
        "pt_jet2": 45,
        "pt_jet3": 35,
        "mean_pnet_jet": 0.65,
        "tight_cuts": True,
    },
    function=cuts_f.hh4b_presel_cuts,
)

hh4b_2b_region = Cut(
    name="hh4b_2b_region",
    params={
        "third_pnet_jet": 0.2605,
        "fourth_pnet_jet": 0.2605,
    },
    function=cuts_f.hh4b_2b_cuts,
)
hh4b_4b_region = Cut(
    name="hh4b_4b_region",
    params={
        "third_pnet_jet": 0.2605,
        "fourth_pnet_jet": 0.2605,
    },
    function=cuts_f.hh4b_4b_cuts,
)

hh4b_signal_region = Cut(
    name="hh4b_signal_region",
    params={
        "radius": "Rhh",
        "radius_min": 0,
        "radius_max": 30,
        },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_control_region = Cut(
    name="hh4b_control_region",
    params={
        "radius": "Rhh",
        "radius_min": 30,
        "radius_max": 55,
        },
    function=cuts_f.hh4b_Rhh_cuts,
)

signal_region_run2 = Cut(
    name="signal_region_run2",
    params={
        "radius": "Rhh_Run2",
        "radius_min": 0,
        "radius_max": 30,
        },
    function=cuts_f.hh4b_Rhh_cuts,
)

control_region_run2 = Cut(
    name="control_region_run2",
    params={
        "radius": "Rhh_Run2",
        "radius_min": 30,
        "radius_max": 55,
        },
    function=cuts_f.hh4b_Rhh_cuts,
)
