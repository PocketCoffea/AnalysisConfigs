import awkward as ak
import sys
from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.lib.deltaR_matching import object_matching

from custom_cut_functions import *
from custom_cuts import *


class HH4bbQuarkMatchingProcessor(BaseProcessorABC):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.dr_min = self.workflow_options["parton_jet_min_dR"]
        self.max_num_jets = self.workflow_options["max_num_jets"]
        self.which_bquark = self.workflow_options["which_bquark"]

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
        # keep only the first 4 jets for the Higgs candidates reconstruction
        self.events["JetGoodHiggs"] = self.events.JetGood[:, : self.max_num_jets]

        self.events["JetGoodHiggsPtOrder"] = self.events.JetGoodHiggs[
            ak.argsort(
                self.events.JetGoodHiggs.ptPnetRegNeutrino, axis=1, ascending=False
            )
        ]


    def do_parton_matching(self, which_bquark):  # -> ak.Array:
        # Select b-quarks at Gen level, coming from H->bb decay
        self.events.GenPart = ak.with_field(
            self.events.GenPart, ak.local_index(self.events.GenPart, axis=1), "index"
        )
        isHiggs = self.events.GenPart.pdgId == 25
        isLast = self.events.GenPart.hasFlags(["isLastCopy"])
        isHard = self.events.GenPart.hasFlags(["fromHardProcess"])
        higgs = self.events.GenPart[isHiggs & isLast & isHard]

        higgs = higgs[ak.num(higgs.childrenIdxG, axis=2) == 2]

        higgs = higgs[ak.argsort(higgs.pt, ascending=False)]
        if which_bquark == "last":
            isB = abs(self.events.GenPart.pdgId) == 5
            bquarks = self.events.GenPart[isB & isLast & isHard]
            bquarks_first = bquarks
            while True:
                b_mother = self.events.GenPart[bquarks_first.genPartIdxMother]
                mask_mother = (abs(b_mother.pdgId) == 5) | ((b_mother.pdgId) == 25)
                bquarks = bquarks[mask_mother]
                bquarks_first = bquarks_first[mask_mother]
                b_mother = b_mother[mask_mother]
                if ak.all((b_mother.pdgId) == 25):
                    break
                bquarks_first = ak.where(
                    abs(b_mother.pdgId) == 5, b_mother, bquarks_first
                )
            provenance = ak.where(
                bquarks_first.genPartIdxMother == higgs.index[:, 0], 1, 2
            )
        elif which_bquark == "first":
            bquarks = ak.flatten(higgs.children, axis=2)
            provenance = ak.where(bquarks.genPartIdxMother == higgs.index[:, 0], 1, 2)
        else:
            raise ValueError(
                "which_bquark for the parton matching must be 'first' or 'last'"
            )

        # Adding the provenance to the quark object
        bquarks = ak.with_field(bquarks, provenance, "provenance")
        self.events["bQuark"] = bquarks

        # Calling our general object_matching function.
        # The output is an awkward array with the shape of the second argument and None where there is no matching.
        # So, calling like this, we will get out an array of matched_quarks with the dimension of the JetGood.
        matched_bquarks_higgs, matched_jets_higgs, deltaR_matched_higgs = (
            object_matching(
                bquarks,
                self.events.JetGoodHiggs,
                dr_min=self.dr_min,
            )
        )
        # matched all jetgood
        matched_bquarks, matched_jets, deltaR_matched = object_matching(
            bquarks,
            self.events.JetGood,
            dr_min=self.dr_min,
        )

        matched_jets_higgs = ak.with_field(
            matched_jets_higgs, matched_bquarks_higgs.provenance, "provenance"
        )
        self.events["JetGoodHiggs"] = ak.with_field(
            self.events.JetGoodHiggs, matched_bquarks_higgs.provenance, "provenance"
        )
        matched_jets = ak.with_field(
            matched_jets, matched_bquarks.provenance, "provenance"
        )
        self.events["JetGood"] = ak.with_field(
            self.events.JetGood, matched_bquarks.provenance, "provenance"
        )

        self.events["bQuarkHiggsMatched"] = ak.with_field(
            matched_bquarks_higgs, deltaR_matched_higgs, "dRMatchedJet"
        )
        self.events["JetGoodHiggsMatched"] = ak.with_field(
            matched_jets_higgs, deltaR_matched_higgs, "dRMatchedJet"
        )
        self.events["bQuarkMatched"] = ak.with_field(
            matched_bquarks, deltaR_matched, "dRMatchedJet"
        )
        self.events["JetGoodMatched"] = ak.with_field(
            matched_jets, deltaR_matched, "dRMatchedJet"
        )
        self.events["JetGoodHiggsMatched"] = ak.with_field(
            self.events.JetGoodHiggsMatched,
            self.events.bQuarkHiggsMatched.pdgId,
            "pdgId",
        )
        self.events["JetGoodMatched"] = ak.with_field(
            self.events.JetGoodMatched,
            self.events.bQuarkMatched.pdgId,
            "pdgId",
        )

    def dummy_provenance(self):
        self.events["JetGoodHiggs"] = ak.with_field(
            self.events.JetGoodHiggs,
            ak.ones_like(self.events.JetGoodHiggs.pt) * -1,
            "provenance"
        )
        self.events["JetGoodHiggsMatched"] = self.events.JetGoodHiggs

        self.events["JetGood"] = ak.with_field(
            self.events.JetGood,
            ak.ones_like(self.events.JetGood.pt) * -1,
            "provenance"
        )
        self.events["JetGoodMatched"] = self.events.JetGood

    def count_objects(self, variation):
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood, axis=1)
        self.events["nMuonGood"] = ak.num(self.events.MuonGood, axis=1)
        self.events["nJetGood"] = ak.num(self.events.JetGood, axis=1)
        self.events["nJetGoodHiggs"] = ak.num(self.events.JetGoodHiggs, axis=1)

    def reconstruct_higgs_candidates(self, matched_jets_higgs):

        jet_higgs1 = matched_jets_higgs[matched_jets_higgs.provenance == 1]
        jet_higgs2 = matched_jets_higgs[matched_jets_higgs.provenance == 2]

        reco_higgs1 = jet_higgs1[:, 0] + jet_higgs1[:, 1]
        reco_higgs2 = jet_higgs2[:, 0] + jet_higgs2[:, 1]
        reco_higgs1 = ak.with_field(reco_higgs1, reco_higgs1.pt, "pt")
        reco_higgs2 = ak.with_field(reco_higgs2, reco_higgs2.pt, "pt")
        reco_higgs1 = ak.with_field(reco_higgs1, reco_higgs1.eta, "eta")
        reco_higgs2 = ak.with_field(reco_higgs2, reco_higgs2.eta, "eta")
        reco_higgs1 = ak.with_field(reco_higgs1, reco_higgs1.phi, "phi")
        reco_higgs2 = ak.with_field(reco_higgs2, reco_higgs2.phi, "phi")
        reco_higgs1 = ak.with_field(reco_higgs1, reco_higgs1.mass, "mass")
        reco_higgs2 = ak.with_field(reco_higgs2, reco_higgs2.mass, "mass")

        return reco_higgs1, reco_higgs2

    def process_extra_after_presel(self, variation) -> ak.Array:
        if self._isMC:
            self.do_parton_matching(which_bquark=self.which_bquark)
            # NOTE:  ak.num counts even the None values, while ak.count counts only the non-None values
            self.events["nbQuarkHiggsMatched"] = ak.num(
                self.events.bQuarkHiggsMatched, axis=1
            )
            self.events["nbQuarkMatched"] = ak.num(self.events.bQuarkMatched, axis=1)

            # collection with the pt regressed without neutrino
            self.events["JetGoodHiggsRegMatched"] = ak.with_field(
                self.events.JetGoodHiggsMatched,
                self.events.JetGoodHiggsMatched.pt
                * (1 - self.events.JetGoodHiggsMatched.rawFactor)
                * self.events.JetGoodHiggsMatched.PNetRegPtRawCorr,
                "pt",
            )
            # collection with the pt regressed with neutrino
            self.events["JetGoodHiggsRegNeutrinoMatched"] = ak.with_field(
                self.events.JetGoodHiggsMatched,
                self.events.JetGoodHiggsMatched.pt
                * (1 - self.events.JetGoodHiggsMatched.rawFactor)
                * self.events.JetGoodHiggsMatched.PNetRegPtRawCorr
                * self.events.JetGoodHiggsMatched.PNetRegPtRawCorrNeutrino,
                "pt",
            )

            # reconstruct the higgs candidates
            self.events["RecoHiggs1"], self.events["RecoHiggs2"] = (
                self.reconstruct_higgs_candidates(self.events.JetGoodHiggsMatched)
            )

            # reconstruct the higgs candidates with the pt regressed without neutrino
            self.events["PNetRegRecoHiggs1"], self.events["PNetRegRecoHiggs2"] = (
                self.reconstruct_higgs_candidates(self.events.JetGoodHiggsRegMatched)
            )

            # reconstruct the higgs candidates with the pt regressed with neutrino
            (
                self.events["PNetRegNeutrinoRecoHiggs1"],
                self.events["PNetRegNeutrinoRecoHiggs2"],
            ) = self.reconstruct_higgs_candidates(
                self.events.JetGoodHiggsRegNeutrinoMatched
            )
        else:
            self.dummy_provenance()


        self.events["nJetGoodHiggsMatched"] = ak.num(
            self.events.JetGoodHiggsMatched, axis=1
        )
        self.events["nJetGoodMatched"] = ak.num(self.events.JetGoodMatched, axis=1)