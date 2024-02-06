import awkward as ak
from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.lib.deltaR_matching import object_matching

from custom_cut_functions import *
from custom_cuts import *


class HH4bPartonMatchingProcessor(BaseProcessorABC):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.dr_min = self.workflow_options["parton_jet_min_dR"]
        self.max_num_jets = self.workflow_options["max_num_jets"]

    def apply_object_preselection(self, variation):
        # super().apply_object_preselection(variation=variation)
        self.events["Jet"] = ak.with_field(
            self.events.Jet,
            self.events.Jet.pt
            * (1 - self.events.Jet.rawFactor)
            * self.events.Jet.PNetRegPtRawCorr
            * self.events.Jet.PNetRegPtRawCorrNeutrino,
            "ptPnetRegNeutrino",
        )
        self.events["JetGood"] = jet_selection_nopu(self.events, "Jet", self.params)
        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
        )
        self.events["MuonGood"] = lepton_selection(self.events, "Muon", self.params)
        # order jet by btag score and keep only the first 4
        self.events["JetGood"] = self.events.JetGood[
            ak.argsort(self.events.JetGood.btagPNetB, axis=1, ascending=False)
        ]
        self.events["JetGoodBTagOrder"] = self.events.JetGood[:, : self.max_num_jets]

        self.events["JetGoodPtOrder"] = self.events.JetGoodBTagOrder[
            ak.argsort(self.events.JetGoodBTagOrder.ptPnetRegNeutrino, axis=1, ascending=False)
        ]

    # def define_common_variables_before_presel(self, variation):

    def do_parton_matching(self):  # -> ak.Array:
        # Select b-quarks at Gen level, coming from H->bb decay

        higgs = self.events.GenPart[
            (self.events.GenPart.pdgId == 25)
            & self.events.GenPart.hasFlags(["fromHardProcess"])
            & self.events.GenPart.hasFlags(["isLastCopy"])
        ]
        higgs = higgs[ak.num(higgs.childrenIdxG, axis=2) == 2]
        #print("pt", higgs.pt[:5])
        #print(bquarks.genPartIdxMother)
        higgs=higgs[ak.argsort(higgs.pt,ascending=False)]
        bquarks = ak.flatten(higgs.children, axis=2)

        #TODO: loop over the children of the bquarks until we find the last copy

        #print("pt", higgs.pt[:5])
        #print(bquarks.genPartIdxMother)

        bquarks_pairs = ak.combinations(bquarks, 2)
        same_higgs = (
            bquarks_pairs["0"].genPartIdxMother == bquarks_pairs["1"].genPartIdxMother
        )
        bquarks_pairs = bquarks_pairs[same_higgs]

        bquarks_pairs_idx = ak.argcombinations(bquarks, 2)
        bquarks_pairs_idx = bquarks_pairs_idx[same_higgs]

        # Get the interpretation
        provenance = ak.to_numpy(ak.zeros_like(bquarks.pdgId))

        for k in [0, 1]:
            provenance[:, bquarks_pairs_idx[:, k]["0"]] = k + 1
            provenance[:, bquarks_pairs_idx[:, k]["1"]] = k + 1

        # Adding the provenance to the quark object
        bquarks = ak.with_field(bquarks, provenance, "provenance")
        self.events["Parton"] = bquarks
        #print(bquarks.provenance)
        #print(bquarks.genPartIdxMother)
        # #print(len(bquarks.provenance[bquarks.provenance == 1]))
        # #print(len(bquarks.provenance[bquarks.provenance == 2]))

        # Calling our general object_matching function.
        # The output is an awkward array with the shape of the second argument and None where there is no matching.
        # So, calling like this, we will get out an array of matched_quarks with the dimension of the JetGood.
        matched_bquarks, matched_jets, deltaR_matched = object_matching(
            bquarks,
            self.events.JetGoodBTagOrder,  # [:, :self.max_num_jets],
            dr_min=self.dr_min,
        )
        matched_bquarks = matched_bquarks[~ak.is_none(matched_bquarks, axis=1)]
        matched_jets = matched_jets[~ak.is_none(matched_jets, axis=1)]
        deltaR_matched = deltaR_matched[~ak.is_none(deltaR_matched, axis=1)]

        # mask and keep only events with exatcly 4 matched b-quarks
        # mask = ak.num(matched_bquarks, axis=1) == 4
        # #print("mask", mask, len(mask))
        # matched_bquarks = matched_bquarks[mask]
        # matched_jets = matched_jets[mask]
        # deltaR_matched = deltaR_matched[mask]

        # #print("dR_matched", deltaR_matched)
        # #print(matched_bquarks.provenance)
        # #print(matched_bquarks.provenance == 1)
        # #print(matched_bquarks.provenance == 2)
        # #print(ak.num(matched_bquarks.provenance[matched_bquarks.provenance == 1]))
        # #print(ak.num(matched_bquarks.provenance[matched_bquarks.provenance == 2]))

        matched_jets = ak.with_field(
            matched_jets, matched_bquarks.provenance, "provenance"
        )
        # #print(matched_jets.provenance)
        # #print(matched_jets.provenance == 1)
        # #print(matched_jets.provenance == 2)

        # #print(ak.sum(matched_jets.provenance[matched_jets.provenance == 1]))
        # #print(ak.sum(matched_jets.provenance[matched_jets.provenance == 2]))

        self.events["PartonMatched"] = ak.with_field(
            matched_bquarks, deltaR_matched, "dRMatchedJet"
        )
        self.events["JetGoodBTagOrderMatched"] = ak.with_field(
            matched_jets, deltaR_matched, "dRMatchedJet"
        )
        # self.events["JetGoodBTagOrderMatched"] = ak.with_field(
        #     self.events.JetGoodBTagOrderMatched,
        #     self.events.PartonMatched.provenance,
        #     "provenance",
        # )
        self.events["JetGoodBTagOrderMatched"] = ak.with_field(
            self.events.JetGoodBTagOrderMatched,
            self.events.PartonMatched.pdgId,
            "pdgId",
        )
        # self.matched_partons_mask = ~ak.is_none(
        #     self.events.JetGoodMatched, axis=1
        # )

    def count_objects(self, variation):
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood, axis=1)
        self.events["nMuonGood"] = ak.num(self.events.MuonGood, axis=1)
        self.events["nJetGood"] = ak.num(self.events.JetGood, axis=1)
        self.events["nJetGoodBTagOrder"] = ak.num(self.events.JetGoodBTagOrder, axis=1)

    def process_extra_after_presel(self, variation) -> ak.Array:
        self.do_parton_matching()
        self.events["nJetGoodBTagOrderMatched"] = ak.num(
            self.events.JetGoodBTagOrderMatched, axis=1
        )
        self.events["nPartonMatched"] = ak.count(self.events.PartonMatched.pt, axis=1)

        matched_bquarks = self.events.PartonMatched
        matched_jets = self.events.JetGoodBTagOrderMatched

        #print("\n matched gen")
        mask_num = ak.num(matched_bquarks, axis=1) == 4
        # replasce false with none
        mask_num = ak.mask(mask_num, mask_num)
        #print("mask_num", mask_num)
        #print("matched_bquarks.provenance == 1", matched_bquarks.provenance == 1)
        #print("matched_bquarks.provenance == 2", matched_bquarks.provenance == 2)
        # compute invariant mass of the two b-quarks matched to the Higgs
        bquark_higgs1 = matched_bquarks[matched_bquarks.provenance == 1]
        bquark_higgs2 = matched_bquarks[matched_bquarks.provenance == 2]
        bquark_higgs1 = bquark_higgs1[mask_num]
        bquark_higgs2 = bquark_higgs2[mask_num]
        #print(bquark_higgs1)
        #print(bquark_higgs2)
        #print(bquark_higgs1.px)
        #print(bquark_higgs2.px)
        #print(bquark_higgs1.provenance)
        #print(bquark_higgs2.provenance)

        #print(bquark_higgs1[:, 0].provenance)
        #print(bquark_higgs1[:, 1].provenance)
        #print(bquark_higgs2[:, 0].provenance)
        #print(bquark_higgs2[:, 1].provenance)

        higgs1 = bquark_higgs1[:, 0] + bquark_higgs1[:, 1]
        higgs2 = bquark_higgs2[:, 0] + bquark_higgs2[:, 1]
        #print("pt", higgs1.pt[:5])
        #print("pt", higgs2.pt[:5])
        higgs1_mass = higgs1.mass
        higgs2_mass = higgs2.mass
        #print(higgs1_mass)
        #print(higgs2_mass)
        self.events["GenHiggs1Mass"] = higgs1_mass
        self.events["GenHiggs2Mass"] = higgs2_mass
        self.events["GenHiggs1Pt"] = higgs1.pt
        self.events["GenHiggs2Pt"] = higgs2.pt

        # compute invariant mass of the two b-jets matched to the Higgs
        #print("\n jet")
        jet_higgs1 = matched_jets[matched_jets.provenance == 1]
        jet_higgs2 = matched_jets[matched_jets.provenance == 2]
        #print("matched_jets.provenance==1", matched_jets.provenance == 1)
        #print("matched_jets.provenance==2", matched_jets.provenance == 2)
        mask_num = ak.num(matched_jets, axis=1) == 4
        # replasce false with none
        mask_num = ak.mask(mask_num, mask_num)
        #print("mask_num", mask_num)
        #print("jet_higgs1", jet_higgs1)
        #print("jet_higgs2", jet_higgs2)
        jet_higgs1 = jet_higgs1[mask_num]
        jet_higgs2 = jet_higgs2[mask_num]
        #print("jet_higgs1", jet_higgs1)
        #print("jet_higgs2", jet_higgs2)
        #print(jet_higgs1.px)
        #print(jet_higgs2.px)
        #print(jet_higgs1.py)
        #print(jet_higgs2.py)
        #print(jet_higgs1.pz)
        #print(jet_higgs2.pz)
        #print(jet_higgs1.provenance)
        #print(jet_higgs2.provenance)
        #print(jet_higgs1[:, 0].provenance)
        #print(jet_higgs1[:, 1].provenance)
        #print(jet_higgs2[:, 0].provenance)
        #print(jet_higgs2[:, 1].provenance)
        reco_higgs1 = jet_higgs1[:, 0] + jet_higgs1[:, 1]
        reco_higgs2 = jet_higgs2[:, 0] + jet_higgs2[:, 1]
        #print(reco_higgs1.px)
        #print(reco_higgs2.px)
        #print(reco_higgs1.py)
        #print(reco_higgs2.py)
        #print(reco_higgs1.pz)
        #print(reco_higgs2.pz)
        #print(reco_higgs1.energy)
        #print(reco_higgs2.energy)

        reco_higgs1_mass = reco_higgs1.mass
        reco_higgs2_mass = reco_higgs2.mass
        #print(reco_higgs1_mass)
        #print(reco_higgs2_mass)
        self.events["RecoHiggs1Mass"] = reco_higgs1_mass
        self.events["RecoHiggs2Mass"] = reco_higgs2_mass
        self.events["RecoHiggs1Pt"] = reco_higgs1.pt
        self.events["RecoHiggs2Pt"] = reco_higgs2.pt

        #print("deltaR", matched_bquarks.dRMatchedJet)
        #print("deltaR", matched_jets.dRMatchedJet)
        #print("deltaEta", matched_bquarks.eta - matched_jets.eta)
        #print("deltaPhi", matched_bquarks.phi - matched_jets.phi)
        #print("deltaPt", matched_bquarks.pt - matched_jets.pt)

        # plot inv mass of higgs from all partons (Even the ones not matched to jets)
        #print("\n all gen")
        bquarks = self.events.Parton

        #print("bquarks.provenance == 1", bquarks.provenance == 1)
        #print("bquarks.provenance == 2", bquarks.provenance == 2)
        # compute invariant mass of the two b-quarks matched to the Higgs
        bquark_higgs1 = bquarks[bquarks.provenance == 1]
        bquark_higgs2 = bquarks[bquarks.provenance == 2]
        #print(bquark_higgs1)
        #print(bquark_higgs2)
        #print(bquark_higgs1.px)
        #print(bquark_higgs2.px)
        #print(bquark_higgs1.provenance)
        #print(bquark_higgs2.provenance)

        #print(bquark_higgs1[:, 0].provenance)
        #print(bquark_higgs1[:, 1].provenance)
        #print(bquark_higgs2[:, 0].provenance)
        #print(bquark_higgs2[:, 1].provenance)

        higgs1 = bquark_higgs1[:, 0] + bquark_higgs1[:, 1]
        higgs2 = bquark_higgs2[:, 0] + bquark_higgs2[:, 1]
        #print(higgs1.px)
        #print(higgs2.px)
        higgs1_mass = higgs1.mass
        higgs2_mass = higgs2.mass
        #print(higgs1_mass)
        #print(higgs2_mass)
        self.events["AllGenHiggs1Mass"] = higgs1_mass
        self.events["AllGenHiggs2Mass"] = higgs2_mass
        self.events["AllGenHiggs1Pt"] = higgs1.pt
        self.events["AllGenHiggs2Pt"] = higgs2.pt

        # invaraint mass applying regression to the jets
        #print("\n reco regressed jet")
        #print("pt", matched_jets.pt)
        #print("px", matched_jets.px)
        #print("py", matched_jets.py)
        #print("pz", matched_jets.pz)
        #print("energy", matched_jets.energy)
        matched_jets = ak.with_field(
            matched_jets,
            matched_jets.pt
            * (1 - matched_jets.rawFactor)
            * matched_jets.PNetRegPtRawCorr,
            "pt",
        )

        #print("pt", matched_jets.pt)
        #print("px", matched_jets.px)
        #print("py", matched_jets.py)
        #print("pz", matched_jets.pz)
        #print("energy", matched_jets.energy)

        jet_higgs1 = matched_jets[matched_jets.provenance == 1]
        jet_higgs2 = matched_jets[matched_jets.provenance == 2]
        #print("matched_jets.provenance==1", matched_jets.provenance == 1)
        #print("matched_jets.provenance==2", matched_jets.provenance == 2)
        mask_num = ak.num(matched_jets, axis=1) == 4
        # replasce false with none
        mask_num = ak.mask(mask_num, mask_num)
        #print("mask_num", mask_num)
        #print("jet_higgs1", jet_higgs1)
        #print("jet_higgs2", jet_higgs2)
        jet_higgs1 = jet_higgs1[mask_num]
        jet_higgs2 = jet_higgs2[mask_num]
        #print("jet_higgs1", jet_higgs1)
        #print("jet_higgs2", jet_higgs2)
        #print(jet_higgs1.px)
        #print(jet_higgs2.px)
        #print(jet_higgs1.py)
        #print(jet_higgs2.py)
        #print(jet_higgs1.pz)
        #print(jet_higgs2.pz)
        #print(jet_higgs1.provenance)
        #print(jet_higgs2.provenance)
        #print(jet_higgs1[:, 0].provenance)
        #print(jet_higgs1[:, 1].provenance)
        #print(jet_higgs2[:, 0].provenance)
        #print(jet_higgs2[:, 1].provenance)
        reco_higgs1 = jet_higgs1[:, 0] + jet_higgs1[:, 1]
        reco_higgs2 = jet_higgs2[:, 0] + jet_higgs2[:, 1]
        #print(reco_higgs1.px)
        #print(reco_higgs2.px)
        #print(reco_higgs1.py)
        #print(reco_higgs2.py)
        #print(reco_higgs1.pz)
        #print(reco_higgs2.pz)
        #print(reco_higgs1.energy)
        #print(reco_higgs2.energy)

        reco_higgs1_mass = reco_higgs1.mass
        reco_higgs2_mass = reco_higgs2.mass
        #print(reco_higgs1_mass)
        #print(reco_higgs2_mass)
        self.events["PNetRegRecoHiggs1Mass"] = reco_higgs1_mass
        self.events["PNetRegRecoHiggs2Mass"] = reco_higgs2_mass
        self.events["PNetRegRecoHiggs1Pt"] = reco_higgs1.pt
        self.events["PNetRegRecoHiggs2Pt"] = reco_higgs2.pt

        #print("\n pnet w/ neutrino")
        matched_jets = ak.with_field(
            matched_jets,
            matched_jets.pt*matched_jets.PNetRegPtRawCorrNeutrino,
            "pt",
        )

        jet_higgs1 = matched_jets[matched_jets.provenance == 1]
        jet_higgs2 = matched_jets[matched_jets.provenance == 2]
        #print("matched_jets.provenance==1", matched_jets.provenance == 1)
        #print("matched_jets.provenance==2", matched_jets.provenance == 2)
        mask_num = ak.num(matched_jets, axis=1) == 4
        # replasce false with none
        mask_num = ak.mask(mask_num, mask_num)
        #print("mask_num", mask_num)
        #print("jet_higgs1", jet_higgs1)
        #print("jet_higgs2", jet_higgs2)
        jet_higgs1 = jet_higgs1[mask_num]
        jet_higgs2 = jet_higgs2[mask_num]
        #print("jet_higgs1", jet_higgs1)
        #print("jet_higgs2", jet_higgs2)
        #print(jet_higgs1.px)
        #print(jet_higgs2.px)
        #print(jet_higgs1.py)
        #print(jet_higgs2.py)
        #print(jet_higgs1.pz)
        #print(jet_higgs2.pz)
        #print(jet_higgs1.provenance)
        #print(jet_higgs2.provenance)
        #print(jet_higgs1[:, 0].provenance)
        #print(jet_higgs1[:, 1].provenance)
        #print(jet_higgs2[:, 0].provenance)
        #print(jet_higgs2[:, 1].provenance)
        reco_higgs1 = jet_higgs1[:, 0] + jet_higgs1[:, 1]
        reco_higgs2 = jet_higgs2[:, 0] + jet_higgs2[:, 1]
        #print(reco_higgs1.px)
        #print(reco_higgs2.px)
        #print(reco_higgs1.py)
        #print(reco_higgs2.py)
        #print(reco_higgs1.pz)
        #print(reco_higgs2.pz)
        #print(reco_higgs1.energy)
        #print(reco_higgs2.energy)

        reco_higgs1_mass = reco_higgs1.mass
        reco_higgs2_mass = reco_higgs2.mass
        #print(reco_higgs1_mass)
        #print(reco_higgs2_mass)
        self.events["PNetRegNeutrinoRecoHiggs1Mass"] = reco_higgs1_mass
        self.events["PNetRegNeutrinoRecoHiggs2Mass"] = reco_higgs2_mass
        self.events["PNetRegNeutrinoRecoHiggs1Pt"] = reco_higgs1.pt
        self.events["PNetRegNeutrinoRecoHiggs2Pt"] = reco_higgs2.pt