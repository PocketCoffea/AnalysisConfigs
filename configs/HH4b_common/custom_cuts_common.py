from collections.abc import Iterable
import numpy as np
import awkward as ak

import custom_cut_functions as cuts_f
from pocket_coffea.lib.cut_definition import Cut
from vbf_matching import mask_efficiency

vbf_hh4b_presel = Cut(
    name="hh4b",
    params={
        "njetgood": 4,
        "njetvbf": 6,
        "pt_jet0": 80,
        "pt_jet1": 60,
        "pt_jet2": 45,
        "pt_jet3": 35,
        "mean_pnet_jet": 0.65,
    },
    function=cuts_f.vbf_hh4b_presel_cuts,
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

semiTight_leadingPt = Cut(
    name="semiTight_leadingPt",
    params={
        "mjj": 400,
        "deltaEta_jj": 3.5,
    },
    function=cuts_f.semiTight_leadingPt,
    )

semiTight_leadingMjj = Cut(
    name="semiTight_leadingMjj",
    params={
        "mjj": 400,
        "deltaEta_jj": 3.5,
    },
    function=cuts_f.semiTight_leadingMjj,
    )

VBF_region = Cut(
    name="VBF",
    params={
        "njet_vbf": 2,
        "delta_eta": 5,
    },
    function=cuts_f.VBF_cuts,
)

VBF_generalSelection_region = Cut(
    name="4b_VBF_genSel",
    params={
        "njet_vbf": 2,
        "pt_VBFjet0": 30,
        "eta_product": 0,
        "mjj": 250,
    },
    function=cuts_f.VBF_generalSelection_cuts,
)

# VBFtight_region = Cut(
#     name="4b_VBFtight",
#     params={
#         "njet_vbf": 2,
#         "eta_product": 0,
#         "mjj": 350,
#     },
#     function=cuts_f.VBFtight_cuts,
# )

# Default parameters dictionary
VBFtight_params = {
    "njet_vbf": 2,
    "eta_product": 0,
    "mjj": 350,
    "pt": 10,
    "eta": 4.7,
    "btag": 0.2605
}

# Different parameters dictionary
no_cuts_params = {
    "njet_vbf": 2,
    "eta_product": 2,
    "mjj": -1,
    "pt": -1,
    "eta": 20,
    "btag": 2
}

def vbf_wrapper(params = VBFtight_params):
    return Cut(
        name="4b_VBFtight",
        params=params,
        function=cuts_f.VBFtight_cuts,
        )

def generate_dictionaries(VBFtight_params, no_cuts_params):
    dict_array = []
    for key in no_cuts_params.keys():
        temp_dict = no_cuts_params.copy()
        temp_dict[key] = VBFtight_params[key]
        dict_array.append(temp_dict)


    return dict_array

# Generate the array of dictionaries
ab = generate_dictionaries(VBFtight_params, no_cuts_params)
print(len(ab))
for i in range(0, len(ab)):
    print(list(no_cuts_params.keys())[i], ab[i], "\n")

qvg_regions = {}
for i in range(5, 10):
    qvg_regions[f'qvg_0{i}_region'] = Cut(
        name=f'qvg0{i}',
        params={
            "qvg_cut": i/10
        },
        function=cuts_f.qvg_cuts,
)

def lepton_selection(events, lepton_flavour, params):
    leptons = events[lepton_flavour]
    cuts = params.object_preselection[lepton_flavour]
    # Requirements on pT and eta
    passes_eta = abs(leptons.eta) < cuts["eta"]
    passes_pt = leptons.pt > cuts["pt"]

    passes_dxy = ak.where(
        abs(leptons.eta) < 1.479,
        leptons.dxy < cuts["dxy_barrel"],
        leptons.dxy < cuts["dxy_endcap"],
    )
    passes_dz = ak.where(
        abs(leptons.eta) < 1.479,
        leptons.dz < cuts["dz_barrel"],
        leptons.dz < cuts["dz_endcap"],
    )

    if lepton_flavour == "Electron":
        # Requirements on SuperCluster eta, isolation and id
        # etaSC = abs(leptons.deltaEtaSC + leptons.eta)
        # passes_SC = np.invert((etaSC >= 1.4442) & (etaSC <= 1.5660))
        passes_iso = leptons.pfRelIso03_all < cuts["iso"]
        passes_id = leptons[cuts["id"]] == True
        good_leptons = (
            passes_eta & passes_pt & passes_iso & passes_dxy & passes_dz & passes_id
        )
        # & passes_SC

    elif lepton_flavour == "Muon":
        # Requirements on isolation and id
        passes_iso = leptons.pfRelIso03_all < cuts["iso"]
        passes_id = leptons[cuts["id"]] == True

        good_leptons = (
            passes_eta & passes_pt & passes_iso & passes_dxy & passes_dz & passes_id
        )

    return leptons[good_leptons]


def jet_selection_nopu(events, jet_type, params):
    jets = events[jet_type]
    cuts = params.object_preselection[jet_type]
    # print(jet_type, cuts)

    # Only jets that are more distant than dr to ALL leptons are tagged as good jets
    # Mask for  jets not passing the preselection
    if "eta_min" in cuts.keys():
        mask_jets = (
            (jets.pt > cuts["pt"])
            & (np.abs(jets.eta) > cuts["eta_min"])
            & (np.abs(jets.eta) < cuts["eta_max"])
            & (jets.jetId >= cuts["jetId"])
            & (jets.btagPNetB >= cuts["btagPNetB"])
        )
    else:
        mask_jets = (
            (jets.pt > cuts["pt"])
            & (np.abs(jets.eta) < cuts["eta"])
            & (jets.jetId >= cuts["jetId"])
            & (jets.btagPNetB >= cuts["btagPNetB"])
        )

    return jets[mask_jets]