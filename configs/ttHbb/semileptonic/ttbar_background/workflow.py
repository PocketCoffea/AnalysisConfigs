import awkward as ak
from pocket_coffea.workflows.tthbb_base_processor import ttHbbBaseProcessor
from pocket_coffea.lib.deltaR_matching import metric_eta, metric_phi

class ttbarBackgroundProcessor(ttHbbBaseProcessor):
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

    def count_objects(self, variation):
        super().count_objects(variation=variation)
        if self._isMC:
            self.events["nGenJetGood"] = ak.num(self.events.GenJetGood)
            self.events["nBGenJetGood"] = ak.num(self.events.BGenJetGood)
            self.events["nCGenJetGood"] = ak.num(self.events.CGenJetGood)
            self.events["nLGenJetGood"] = ak.num(self.events.LGenJetGood)
