import awkward as ak
from pocket_coffea.workflows.tthbb_base_processor import ttHbbBaseProcessor
from pocket_coffea.lib.deltaR_matching import metric_eta, metric_phi
from pocket_coffea.lib.deltaR_matching import object_matching
from pocket_coffea.lib.parton_provenance import get_partons_provenance_ttHbb, get_partons_provenance_ttbb4F, get_partons_provenance_tt5F

class ttbarBackgroundProcessor(ttHbbBaseProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.dr_min = self.workflow_options["parton_jet_min_dR"]

    @classmethod
    def available_variations(cls):
        vars = super().available_variations()
        variations_sf_ele_trigger = ["stat", "pileup", "era", "ht"]
        available_sf_ele_trigger_variations = [f"sf_ele_trigger_{v}" for v in variations_sf_ele_trigger]
        variations_sf_btag = ["hf", "lf", "hfstats1", "hfstats2", "lfstats1", "lfstats2", "cferr1", "cferr2"]
        available_sf_btag_variations = [f"sf_btag_{v}" for v in variations_sf_btag]
        vars.update(available_sf_ele_trigger_variations)
        vars.update(available_sf_btag_variations)
        return vars

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation=variation)
        if self._isMC:
            # Apply the GenJet acceptance cuts
            mask_pt = self.events["GenJet"].pt > 20.
            mask_eta = abs(self.events["GenJet"].eta) < 2.4
            mask_acceptance = mask_pt & mask_eta
            # Ghost-hadron matching
            mask_b = self.events["GenJet"].hadronFlavour == 5
            mask_c = self.events["GenJet"].hadronFlavour == 4
            mask_l = self.events["GenJet"].hadronFlavour == 0

            # Build new GenJet collections split by flavour
            self.events["GenJetGood"] = self.events.GenJet[mask_acceptance]
            self.events["BGenJetGood"] = self.events.GenJet[mask_acceptance & mask_b]
            self.events["CGenJetGood"] = self.events.GenJet[mask_acceptance & mask_c]
            self.events["LGenJetGood"] = self.events.GenJet[mask_acceptance & mask_l]

    def define_common_variables_after_presel(self, variation):
        super().define_common_variables_before_presel(variation=variation)

        # Compute deltaR(b, b) of all possible b-jet pairs.
        # We require deltaR > 0 to exclude the deltaR between the jets with themselves
        deltaR = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"]), axis=2)
        deltaEta = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"], metric=metric_eta), axis=2)
        deltaPhi = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"], metric=metric_phi), axis=2)
        deltaR = deltaR[deltaR > 0.]
        deltaEta = deltaEta[deltaEta > 0.]
        deltaPhi = deltaPhi[deltaPhi > 0.]

        # Get the deltaR with no possibility of repetition of identical b-jet pairs

        # Get all the possible combinations of b-jet pairs
        pairs = ak.argcombinations(self.events["BJetGood"], 2, axis=1)
        b1 = self.events.BJetGood[pairs.slot0]
        b2 = self.events.BJetGood[pairs.slot1]

        # Compute deltaR between the pairs
        deltaR_unique = b1.delta_r(b2)
        idx_pairs_sorted = ak.argsort(deltaR_unique, axis=1)
        pairs_sorted = pairs[idx_pairs_sorted]

        # Compute the minimum deltaR(b, b), deltaEta(b, b), deltaPhi(b, b) and the invariant mass of the closest b-jet pair
        self.events["deltaRbb_min"] = ak.min(deltaR, axis=1)
        self.events["deltaEtabb_min"] = ak.min(deltaEta, axis=1)
        self.events["deltaPhibb_min"] = ak.min(deltaPhi, axis=1)
        self.events["mbb"] = (self.events.BJetGood[pairs_sorted.slot0] + self.events.BJetGood[pairs_sorted.slot1]).mass

    def do_parton_matching(self) -> ak.Array:
        # Selects quarks at LHE level
        isOutgoing = self.events.LHEPart.status == 1
        isParton = (abs(self.events.LHEPart.pdgId) < 6) | (self.events.LHEPart.pdgId == 21)
        quarks = self.events.LHEPart[isOutgoing & isParton]

        # Get the interpretation
        if self._sample == "TTbbSemiLeptonic":
            prov = get_partons_provenance_ttbb4F(
                ak.Array(quarks.pdgId, behavior={}), ak.ArrayBuilder()
            ).snapshot()
        elif self._sample == "TTToSemiLeptonic":
            prov = get_partons_provenance_tt5F(
                ak.Array(quarks.pdgId, behavior={}), ak.ArrayBuilder()
            ).snapshot()
        else:
            prov = -1 * ak.ones_like(quarks)

        # Adding the provenance to the quark object
        quarks = ak.with_field(quarks, prov, "provenance")

        # Calling our general object_matching function.
        # The output is an awkward array with the shape of the second argument and None where there is no matching.
        # So, calling like this, we will get out an array of matched_quarks with the dimension of the JetGood.
        matched_quarks, matched_genjets, deltaR_matched = object_matching(
            quarks, self.events.GenJetGood, dr_min=self.dr_min
        )

        # Saving leptons and neutrino parton level
        self.events["LeptonParton"] = self.events.LHEPart[
            (self.events.LHEPart.status == 1)
            & (abs(self.events.LHEPart.pdgId) > 10)
            & (abs(self.events.LHEPart.pdgId) < 15)
        ]

        self.events["Parton"] = quarks
        self.events["PartonMatched"] = ak.with_field(
            matched_quarks, deltaR_matched, "dRMatchedGenJet"
        )
        self.events["GenJetGoodMatched"] = ak.with_field(
            matched_genjets, deltaR_matched, "dRMatchedGenJet"
        )
        self.matched_partons_mask = ~ak.is_none(self.events.GenJetGoodMatched, axis=1)

    def count_partons(self):
        self.events["nParton"] = ak.num(self.events.Parton, axis=1)
        self.events["nPartonMatched"] = ak.count(
            self.events.PartonMatched.pt, axis=1
        )  # use count since we have None

    def count_additional_jets(self):
        # Mask to tag additional b-jets: we require gen-level b-jets that are matched to a quark not originated from a top decay
        is_additional_bjet = ak.fill_none(
            (self.events.GenJetGoodMatched.hadronFlavour == 5) &
            (self.events.PartonMatched.provenance != 2) &
            (self.events.PartonMatched.provenance != 3),
            False
        )
        # Mask to tag additional c-jets: we require gen-level c-jets that are matched to a quark not originated from a top decay
        is_additional_cjet = ak.fill_none(
            (self.events.GenJetGoodMatched.hadronFlavour == 4) &
            (self.events.PartonMatched.provenance != 2) &
            (self.events.PartonMatched.provenance != 3),
            False
        )
        self.events["BGenJetGoodExtra"] = self.events.GenJetGoodMatched[is_additional_bjet]
        self.events["CGenJetGoodExtra"] = self.events.GenJetGoodMatched[is_additional_cjet]
        self.events["nBGenJetGoodExtra"] = ak.num(self.events.BGenJetGoodExtra)
        self.events["nCGenJetGoodExtra"] = ak.num(self.events.CGenJetGoodExtra)

    def count_objects(self, variation):
        super().count_objects(variation=variation)
        if self._isMC:
            self.events["nGenJetGood"] = ak.num(self.events.GenJetGood)
            self.events["nBGenJetGood"] = ak.num(self.events.BGenJetGood)
            self.events["nCGenJetGood"] = ak.num(self.events.CGenJetGood)
            self.events["nLGenJetGood"] = ak.num(self.events.LGenJetGood)

    def process_extra_after_presel(self, variation) -> ak.Array:
        if self._isMC & (self._sample in ["TTbbSemiLeptonic", "TTToSemiLeptonic"]):
            self.do_parton_matching()
            self.count_partons()
            self.count_additional_jets()
