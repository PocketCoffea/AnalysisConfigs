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
            ak.argsort(self.events.JetGoodHiggs.ptPnetRegNeutrino, axis=1, ascending=False)
        ]

    # def define_common_variables_before_presel(self, variation):

    def do_parton_matching(self, which_bquark):  # -> ak.Array:
        # Select b-quarks at Gen level, coming from H->bb decay
        self.events.GenPart=ak.with_field(self.events.GenPart, ak.local_index(self.events.GenPart, axis=1), "index")
        #print("num_events", len(self.events.GenPart))
        isHiggs = self.events.GenPart.pdgId == 25
        isLast = self.events.GenPart.hasFlags(["isLastCopy"])
        isHard = self.events.GenPart.hasFlags(["fromHardProcess"])
        higgs = self.events.GenPart[
            isHiggs & isLast & isHard
        ]


        higgs = higgs[ak.num(higgs.childrenIdxG, axis=2) == 2]

        higgs=higgs[ak.argsort(higgs.pt,ascending=False)]
        num_ev=5
        if which_bquark == "last":
            isB=abs(self.events.GenPart.pdgId) == 5
            bquarks = self.events.GenPart[isB & isLast & isHard]
            #print("bquarks: ", "pdg", bquarks[:num_ev].pdgId, "mother_idx",bquarks[:num_ev].genPartIdxMother, "pt", bquarks[:num_ev].pt)
            bquarks_first = bquarks
            while True:
                #print("\nloop")
                b_mother = self.events.GenPart[bquarks_first.genPartIdxMother]
                mask_mother=(abs(b_mother.pdgId) == 5) | (abs(b_mother.pdgId) == 25)
                #print("mask_mother", mask_mother)
                bquarks=bquarks[mask_mother]
                bquarks_first=bquarks_first[mask_mother]
                b_mother = b_mother[mask_mother]
                #print("old: ", "pdg", bquarks_first[:num_ev].pdgId, "mother_idx",bquarks_first[:num_ev].genPartIdxMother, "pt", bquarks_first[:num_ev].pt)
                #print("mother: ", "pdg", b_mother[:num_ev].pdgId, "mother_idx",b_mother[:num_ev].genPartIdxMother, "pt", b_mother[:num_ev].pt)
                # for k in range(len(b_mother)):
                #     if ak.any((abs(b_mother.pdgId) != 5) & (abs(b_mother.pdgId) != 25)):
                #         #print("loop", b_mother[k].pdgId, b_mother[k].genPartIdxMother, b_mother[k].pt)
                #print(abs(b_mother.pdgId) != 25, len(abs(b_mother.pdgId) != 25))
                #print(b_mother[abs(b_mother.pdgId) != 25].pdgId, len(b_mother[abs(b_mother.pdgId) != 25]))
                if ak.all(abs(b_mother.pdgId) == 25):
                    break
                bquarks_first = ak.where(abs(b_mother.pdgId) == 5, b_mother, bquarks_first)
                #print("new: ", "pdg", bquarks_first[:num_ev].pdgId, "mother_idx",bquarks_first[:num_ev].genPartIdxMother, "pt", bquarks_first[:num_ev].pt)
            provenance = ak.where(bquarks_first.genPartIdxMother == higgs.index[:,0], 1, 2)
            #print("provenance", provenance[:num_ev])
        elif which_bquark == "first":
            bquarks = ak.flatten(higgs.children, axis=2)
            provenance = ak.where(bquarks.genPartIdxMother == higgs.index[:,0], 1, 2)
        else:
            raise ValueError("which_bquark for the parton matching must be 'first' or 'last'")

        #print("\nhiggs", higgs.pt[:num_ev], higgs.index[:num_ev])
        #print("bquarks", bquarks.pt[:num_ev], bquarks.genPartIdxMother[:num_ev])
        #print("provenance", provenance[:num_ev])


        # #print("pt", higgs.pt[:5])
        # #print(bquarks.genPartIdxMother)

        # bquarks_pairs = ak.combinations(bquarks, 2)
        # same_higgs = (
        #     bquarks_pairs["0"].genPartIdxMother == bquarks_pairs["1"].genPartIdxMother
        # )
        # bquarks_pairs = bquarks_pairs[same_higgs]

        # bquarks_pairs_idx = ak.argcombinations(bquarks, 2)
        # bquarks_pairs_idx = bquarks_pairs_idx[same_higgs]

        # # Get the interpretation
        # provenance = ak.to_numpy(ak.zeros_like(bquarks.pdgId))

        # for k in [0, 1]:
        #     provenance[:, bquarks_pairs_idx[:, k]["0"]] = k + 1
        #     provenance[:, bquarks_pairs_idx[:, k]["1"]] = k + 1

        # Adding the provenance to the quark object
        bquarks = ak.with_field(bquarks, provenance, "provenance")
        self.events["bQuark"] = bquarks
        #print(bquarks.provenance[:num_ev])
        #print(bquarks.pt[:num_ev])
        #print(bquarks.genPartIdxMother[:num_ev])
        # #print(len(bquarks.provenance[bquarks.provenance == 1]))
        # #print(len(bquarks.provenance[bquarks.provenance == 2]))

        # Calling our general object_matching function.
        # The output is an awkward array with the shape of the second argument and None where there is no matching.
        # So, calling like this, we will get out an array of matched_quarks with the dimension of the JetGood.
        matched_bquarks_higgs, matched_jets_higgs, deltaR_matched_higgs = object_matching(
            bquarks,
            self.events.JetGoodHiggs,
            dr_min=self.dr_min,
        )
        # matched all jetgood
        matched_bquarks, matched_jets, deltaR_matched = object_matching(
            bquarks,
            self.events.JetGood,
            dr_min=self.dr_min,
        )

        #TODO: remove none
        # matched_bquarks_higgs = matched_bquarks_higgs[~ak.is_none(matched_bquarks_higgs, axis=1)]
        # matched_jets_higgs = matched_jets_higgs[~ak.is_none(matched_jets_higgs, axis=1)]
        # deltaR_matched_higgs = deltaR_matched_higgs[~ak.is_none(deltaR_matched_higgs, axis=1)]
        # matched_bquarks = matched_bquarks[~ak.is_none(matched_bquarks, axis=1)]
        # matched_jets = matched_jets[~ak.is_none(matched_jets, axis=1)]
        # deltaR_matched = deltaR_matched[~ak.is_none(deltaR_matched, axis=1)]



        # mask and keep only events with exatcly 4 matched b-quarks
        # mask = ak.num(matched_bquarks_higgs, axis=1) == 4
        # #print("mask", mask, len(mask))
        # matched_bquarks_higgs = matched_bquarks_higgs[mask]
        # matched_jets_higgs = matched_jets_higgs[mask]
        # deltaR_matched_higgs = deltaR_matched_higgs[mask]

        # #print("dR_matched", deltaR_matched_higgs)
        # #print(matched_bquarks_higgs.provenance)
        # #print(matched_bquarks_higgs.provenance == 1)
        # #print(matched_bquarks_higgs.provenance == 2)
        # #print(ak.num(matched_bquarks_higgs.provenance[matched_bquarks_higgs.provenance == 1]))
        # #print(ak.num(matched_bquarks_higgs.provenance[matched_bquarks_higgs.provenance == 2]))

        matched_jets_higgs = ak.with_field(
            matched_jets_higgs, matched_bquarks_higgs.provenance, "provenance"
        )
        matched_jets= ak.with_field(
            matched_jets, matched_bquarks.provenance, "provenance"
        )
        # #print(matched_jets_higgs.provenance)
        # #print(matched_jets_higgs.provenance == 1)
        # #print(matched_jets_higgs.provenance == 2)

        # #print(ak.sum(matched_jets_higgs.provenance[matched_jets_higgs.provenance == 1]))
        # #print(ak.sum(matched_jets_higgs.provenance[matched_jets_higgs.provenance == 2]))

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

        # self.matched_partons_mask = ~ak.is_none(
        #     self.events.JetGoodMatched, axis=1
        # )

    def count_objects(self, variation):
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood, axis=1)
        self.events["nMuonGood"] = ak.num(self.events.MuonGood, axis=1)
        self.events["nJetGood"] = ak.num(self.events.JetGood, axis=1)
        self.events["nJetGoodHiggs"] = ak.num(self.events.JetGoodHiggs, axis=1)

    def process_extra_after_presel(self, variation) -> ak.Array:
        self.do_parton_matching(which_bquark=self.which_bquark)
        self.events["nJetGoodHiggsMatched"] = ak.num(
            self.events.JetGoodHiggsMatched, axis=1
        )
        self.events["nJetGoodMatched"] = ak.num(
            self.events.JetGoodMatched, axis=1
        )
        self.events["nbQuarkHiggsMatched"] = ak.num(
            self.events.bQuarkHiggsMatched, axis=1
        )
        self.events["nbQuarkMatched"] = ak.num(
            self.events.bQuarkMatched, axis=1
        )


        '''matched_bquarks_higgs = self.events.bQuarkHiggsMatched
        matched_jets_higgs = self.events.JetGoodHiggsMatched

        #print("\n matched gen")
        mask_num = ak.num(matched_bquarks_higgs, axis=1) == 4
        # replasce false with none
        mask_num = ak.mask(mask_num, mask_num)
        #print("mask_num", mask_num)
        #print("matched_bquarks_higgs.provenance == 1", matched_bquarks_higgs.provenance == 1)
        #print("matched_bquarks_higgs.provenance == 2", matched_bquarks_higgs.provenance == 2)
        # compute invariant mass of the two b-quarks matched to the Higgs
        bquark_higgs1 = matched_bquarks_higgs[matched_bquarks_higgs.provenance == 1]
        bquark_higgs2 = matched_bquarks_higgs[matched_bquarks_higgs.provenance == 2]
        bquark_higgs1 = bquark_higgs1[mask_num]
        bquark_higgs2 = bquark_higgs2[mask_num]

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
        jet_higgs1 = matched_jets_higgs[matched_jets_higgs.provenance == 1]
        jet_higgs2 = matched_jets_higgs[matched_jets_higgs.provenance == 2]
        #print("matched_jets_higgs.provenance==1", matched_jets_higgs.provenance == 1)
        #print("matched_jets_higgs.provenance==2", matched_jets_higgs.provenance == 2)
        mask_num = ak.num(matched_jets_higgs, axis=1) == 4
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

        #print("deltaR", matched_bquarks_higgs.dRMatchedJet)
        #print("deltaR", matched_jets_higgs.dRMatchedJet)
        #print("deltaEta", matched_bquarks_higgs.eta - matched_jets_higgs.eta)
        #print("deltaPhi", matched_bquarks_higgs.phi - matched_jets_higgs.phi)
        #print("deltaPt", matched_bquarks_higgs.pt - matched_jets_higgs.pt)



        # invaraint mass applying regression to the jets
        #print("\n reco regressed jet")
        #print("pt", matched_jets_higgs.pt)
        #print("px", matched_jets_higgs.px)
        #print("py", matched_jets_higgs.py)
        #print("pz", matched_jets_higgs.pz)
        #print("energy", matched_jets_higgs.energy)
        matched_jets_higgs = ak.with_field(
            matched_jets_higgs,
            matched_jets_higgs.pt
            * (1 - matched_jets_higgs.rawFactor)
            * matched_jets_higgs.PNetRegPtRawCorr,
            "pt",
        )

        #print("pt", matched_jets_higgs.pt)
        #print("px", matched_jets_higgs.px)
        #print("py", matched_jets_higgs.py)
        #print("pz", matched_jets_higgs.pz)
        #print("energy", matched_jets_higgs.energy)

        jet_higgs1 = matched_jets_higgs[matched_jets_higgs.provenance == 1]
        jet_higgs2 = matched_jets_higgs[matched_jets_higgs.provenance == 2]
        #print("matched_jets_higgs.provenance==1", matched_jets_higgs.provenance == 1)
        #print("matched_jets_higgs.provenance==2", matched_jets_higgs.provenance == 2)
        mask_num = ak.num(matched_jets_higgs, axis=1) == 4
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
        matched_jets_higgs = ak.with_field(
            matched_jets_higgs,
            matched_jets_higgs.pt*matched_jets_higgs.PNetRegPtRawCorrNeutrino,
            "pt",
        )

        jet_higgs1 = matched_jets_higgs[matched_jets_higgs.provenance == 1]
        jet_higgs2 = matched_jets_higgs[matched_jets_higgs.provenance == 2]
        #print("matched_jets_higgs.provenance==1", matched_jets_higgs.provenance == 1)
        #print("matched_jets_higgs.provenance==2", matched_jets_higgs.provenance == 2)
        mask_num = ak.num(matched_jets_higgs, axis=1) == 4
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
        self.events["PNetRegNeutrinoRecoHiggs2Pt"] = reco_higgs2.pt'''