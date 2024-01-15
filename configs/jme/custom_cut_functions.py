import awkward as ak
import numpy as np
from pocket_coffea.lib.cut_definition import Cut

from params.binning import *


def jet_selection_nopu(events, jet_type, params, leptons_collection=""):
    jets = events[jet_type]
    cuts = params.object_preselection[jet_type]
    # Only jets that are more distant than dr to ALL leptons are tagged as good jets
    # Mask for  jets not passing the preselection
    mask_presel = (
        (jets.pt > cuts["pt"])
        & (np.abs(jets.eta) < cuts["eta"])
        # & (jets.jetId >= cuts["jetId"])
    )
    # Lepton cleaning
    if leptons_collection != "":
        dR_jets_lep = jets.metric_table(events[leptons_collection])
        mask_lepton_cleaning = ak.prod(dR_jets_lep > cuts["dr_lepton"], axis=2) == 1

    mask_good_jets = mask_presel  # & mask_lepton_cleaning


    return jets[mask_good_jets], mask_good_jets

def genjet_selection_flavsplit(events, jet_type, flavs):
    jets = events[jet_type]
    mask_flav = jets.partonFlavour == flavs if type(flavs) == int else ak.any([jets.partonFlavour == flav for flav in flavs], axis=0)
    # mask_flav = ak.any([jets.partonFlavour == flav for flav in flavs], axis=0)
    mask_flav = ak.mask(mask_flav, mask_flav)
    return jets[mask_flav]
