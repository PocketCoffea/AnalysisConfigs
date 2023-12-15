import awkward as ak
import numpy as np
from pocket_coffea.lib.cut_definition import Cut

def dimuon(events, params, year, sample, **kwargs):

    # Masks for same-flavor (SF) and opposite-sign (OS)
    SF = ((events.nMuonGood == 2) & (events.nElectronGood == 0))
    OS = events.ll.charge == 0

    mask = (
        (events.nLeptonGood == 2)
        & (ak.firsts(events.MuonGood.pt) > params["pt_leading_muon"])
        & OS & SF
        & (events.ll.mass > params["mll"]["low"])
        & (events.ll.mass < params["mll"]["high"])
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

dimuon_presel = Cut(
    name="dilepton",
    params={
        "pt_leading_muon": 25,
        "mll": {'low': 25, 'high': 2000},
    },
    function=dimuon,
)

def jet_selection_nopu(events, jet_type, params, leptons_collection=""):

    jets = events[jet_type]
    cuts = params.object_preselection[jet_type]
    # Only jets that are more distant than dr to ALL leptons are tagged as good jets
    # Mask for  jets not passing the preselection
    mask_presel = (
        (jets.pt > cuts["pt"])
        & (np.abs(jets.eta) < cuts["eta"])
        & (jets.jetId >= cuts["jetId"])
    )
    # Lepton cleaning
    if leptons_collection != "":
        dR_jets_lep = jets.metric_table(events[leptons_collection])
        mask_lepton_cleaning = ak.prod(dR_jets_lep > cuts["dr_lepton"], axis=2) == 1

    if jet_type == "Jet":
        mask_good_jets = mask_presel #& mask_lepton_cleaning

    elif jet_type == "FatJet":
        # Apply the msd and preselection cuts
        mask_msd = events.FatJet.msoftdrop > cuts["msd"]
        mask_good_jets = mask_presel & mask_msd

    return jets[mask_good_jets], mask_good_jets