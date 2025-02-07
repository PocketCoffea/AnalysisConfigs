from . import custom_cut_functions_common as cuts_f
from pocket_coffea.lib.cut_definition import Cut


hh4b_presel = Cut(
    name="hh4b",
    params={
        "njet": 4,
        "pt_jet0": 80,
        "pt_jet1": 60,
        "pt_jet2": 45,
        "pt_jet3": 35,
        "mean_pnet_jet": 0.65,
        "tight_cuts": False,
        # "third_pnet_jet": 0.2605,
        # "fourth_pnet_jet": 0.2605,
    },
    function=cuts_f.hh4b_presel_cuts,
)

hh4b_presel_tight = Cut(
    name="hh4b",
    params={
        "njet": 4,
        "pt_jet0": 80,
        "pt_jet1": 60,
        "pt_jet2": 45,
        "pt_jet3": 35,
        "mean_pnet_jet": 0.65,
        "tight_cuts": True,
        # "third_pnet_jet": 0.2605,
        # "fourth_pnet_jet": 0.2605,
    },
    function=cuts_f.hh4b_presel_cuts,
)

hh4b_2b_region = Cut(
    name="hh4b",
    params={
        "third_pnet_jet": 0.2605,
        "fourth_pnet_jet": 0.2605,
    },
    function=cuts_f.hh4b_2b_cuts,
)
hh4b_4b_region = Cut(
    name="hh4b",
    params={
        "third_pnet_jet": 0.2605,
        "fourth_pnet_jet": 0.2605,
    },
    function=cuts_f.hh4b_4b_cuts,
)

hh4b_signal_region = Cut(
    name="hh4b",
    params={
        "radius": "Rhh",
        "radius_min": 0,
        "radius_max": 30,
        },
    function=cuts_f.hh4b_Rhh_cuts,
)

hh4b_control_region = Cut(
    name="hh4b",
    params={
        "radius": "Rhh",
        "radius_min": 30,
        "radius_max": 55,
        },
    function=cuts_f.hh4b_Rhh_cuts,
)

run2_signal_region = Cut(
    name="hh4b",
    params={
        "radius": "Rhh_Run2",
        "radius_min": 0,
        "radius_max": 30,
        },
    function=cuts_f.hh4b_Rhh_cuts,
)

run2_control_region = Cut(
    name="hh4b",
    params={
        "radius": "Rhh_Run2",
        "radius_min": 30,
        "radius_max": 55,
        },
    function=cuts_f.hh4b_Rhh_cuts,
)
