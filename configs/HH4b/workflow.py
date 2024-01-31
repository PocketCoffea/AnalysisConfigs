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
        self.events["JetGood"] = jet_selection_nopu(self.events, "Jet", self.params)
        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
        )
        self.events["MuonGood"] = lepton_selection(self.events, "Muon", self.params)
        self.events["JetGoodBtagOrdered"] = self.events.JetGood[
            ak.argsort(self.events.JetGood.btagPNetB, axis=1, ascending=False)
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
        bquarks = ak.flatten(higgs.children, axis=2)

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
        print(bquarks.provenance)
        print(len(bquarks.provenance[bquarks.provenance == 1]))
        print(len(bquarks.provenance[bquarks.provenance == 2]))

        # Calling our general object_matching function.
        # The output is an awkward array with the shape of the second argument and None where there is no matching.
        # So, calling like this, we will get out an array of matched_quarks with the dimension of the JetGood.
        matched_bquarks, matched_jets, deltaR_matched = object_matching(
            bquarks,
            self.events.JetGoodBtagOrdered[:, :self.max_num_jets],
            dr_min=self.dr_min,
        )
        matched_bquarks = matched_bquarks[~ak.is_none(matched_bquarks, axis=1)]
        matched_jets = matched_jets[~ak.is_none(matched_jets, axis=1)]
        deltaR_matched = deltaR_matched[~ak.is_none(deltaR_matched, axis=1)]

        # mask and keep only events with exatcly 4 matched b-quarks
        # mask = ak.num(matched_bquarks, axis=1) == 4
        # print("mask", mask, len(mask))
        # matched_bquarks = matched_bquarks[mask]
        # matched_jets = matched_jets[mask]
        # deltaR_matched = deltaR_matched[mask]

        print("dR_matched", deltaR_matched)
        print(matched_bquarks.provenance)
        print(matched_bquarks.provenance == 1)
        print(matched_bquarks.provenance == 2)
        print(ak.num(matched_bquarks.provenance[matched_bquarks.provenance == 1]))
        print(ak.num(matched_bquarks.provenance[matched_bquarks.provenance == 2]))

        matched_jets = ak.with_field(
            matched_jets, matched_bquarks.provenance, "provenance"
        )
        print(matched_jets.provenance)
        print(matched_jets.provenance == 1)
        print(matched_jets.provenance == 2)

        print(ak.sum(matched_jets.provenance[matched_jets.provenance == 1]))
        print(ak.sum(matched_jets.provenance[matched_jets.provenance == 2]))


        self.events["PartonMatched"] = ak.with_field(
            matched_bquarks, deltaR_matched, "dRMatchedJet"
        )
        self.events["JetGoodBtagOrderedMatched"] = ak.with_field(
            matched_jets, deltaR_matched, "dRMatchedJet"
        )
        # self.events["JetGoodBtagOrderedMatched"] = ak.with_field(
        #     self.events.JetGoodBtagOrderedMatched,
        #     self.events.PartonMatched.provenance,
        #     "provenance",
        # )
        self.events["JetGoodBtagOrderedMatched"] = ak.with_field(
            self.events.JetGoodBtagOrderedMatched,
            self.events.PartonMatched.pdgId,
            "pdgId",
        )
        # self.matched_partons_mask = ~ak.is_none(
        #     self.events.JetGoodBtagOrderedMatched, axis=1
        # )

    def count_objects(self, variation):
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood, axis=1)
        self.events["nMuonGood"] = ak.num(self.events.MuonGood, axis=1)
        self.events["nJetGood"] = ak.num(self.events.JetGood, axis=1)
        self.events["nJetGoodBtagOrdered"] = ak.num(
            self.events.JetGoodBtagOrdered, axis=1
        )

    def process_extra_after_presel(self, variation) -> ak.Array:
        self.do_parton_matching()
        self.events["nJetGoodBtagOrderedMatched"] = ak.num(
            self.events.JetGoodBtagOrderedMatched, axis=1
        )
        self.events["nPartonMatched"] = ak.count(self.events.PartonMatched.pt, axis=1)

        # TODO: invariant mass of the two b-quarks matched to the Higgs
