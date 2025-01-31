import awkward as ak
from pocket_coffea.workflows.tthbb_base_processor import ttHbbBaseProcessor
from pocket_coffea.lib.objects import btagging, met_xy_correction
from pocket_coffea.lib.deltaR_matching import metric_eta, metric_phi
from pocket_coffea.lib.deltaR_matching import object_matching
from pocket_coffea.lib.parton_provenance import get_partons_provenance_ttHbb, get_partons_provenance_ttbb4F, get_partons_provenance_tt5F
import configs.ttHbb.semileptonic.common.lib.jet_parton_matching  as jet_parton_matching

class ttHbbPartonMatchingProcessorFull(ttHbbBaseProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.dr_max = self.workflow_options["parton_jet_max_dR"]
        self.dr_max_postfsr = self.workflow_options.get("parton_jet_max_dR_postfsr", 1.)

    @classmethod
    def available_variations(cls):
        vars = super().available_variations()
        variations_sf_ele_trigger = ["stat", "pileup", "era", "ht"]
        available_sf_ele_trigger_variations = [f"sf_ele_trigger_{v}" for v in variations_sf_ele_trigger]
        variations_sf_btag = ["hf", "lf", "hfstats1", "hfstats2", "lfstats1", "lfstats2", "cferr1", "cferr2"]
        available_sf_btag_variations = [f"sf_btag_{v}" for v in variations_sf_btag]
        vars = vars + available_sf_ele_trigger_variations + available_sf_btag_variations
        return vars

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation=variation)

        # MET xy correction
        met_pt_corr, met_phi_corr = met_xy_correction(self.params, self.events, "MET", self._year, self._era)

        # Overwrite the MET collection with the corrected MET
        self.events["MET"] = ak.with_field(
            self.events.MET,
            met_pt_corr,
            "pt"
        )
        self.events["MET"] = ak.with_field(
            self.events.MET,
            met_phi_corr,
            "phi"
        )

        self.events["LightJetGood"] = btagging(
            self.events["JetGood"],
            self.params.btagging.working_point[self._year],
            wp=self.params.object_preselection.Jet["btag"]["wp"],
            veto=True
        )

    def define_common_variables_before_presel(self, variation):
        super().define_common_variables_before_presel(variation=variation)

        # Compute the scalar sum of the transverse momenta of the b-jets and light jets
        self.events["BJetGood_Ht"] = ak.sum(abs(self.events.BJetGood.pt), axis=1)
        self.events["LightJetGood_Ht"] = ak.sum(abs(self.events.LightJetGood.pt), axis=1)

    def define_common_variables_after_presel(self, variation):
        super().define_common_variables_before_presel(variation=variation)

        # Compute the `is_electron` flag for LeptonGood
        self.events["LeptonGood"] = ak.with_field(
            self.events.LeptonGood,
            ak.values_astype(self.events.LeptonGood.pdgId == 11, bool),
            "is_electron"
        )

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

        # Compute the minimum deltaR(b, b), deltaEta(b, b), deltaPhi(b, b)
        self.events["deltaRbb_min"] = ak.min(deltaR, axis=1)
        self.events["deltaEtabb_min"] = ak.min(deltaEta, axis=1)
        self.events["deltaPhibb_min"] = ak.min(deltaPhi, axis=1)

        # Compute the invariant mass of the closest b-jet pair, the minimum and maximum invariant mass of all b-jet pairs
        mbb = (self.events.BJetGood[pairs_sorted.slot0] + self.events.BJetGood[pairs_sorted.slot1]).mass
        ptbb = (self.events.BJetGood[pairs_sorted.slot0] + self.events.BJetGood[pairs_sorted.slot1]).pt
        htbb = self.events.BJetGood[pairs_sorted.slot0].pt + self.events.BJetGood[pairs_sorted.slot1].pt
        self.events["mbb_closest"] = mbb[:,0]
        self.events["mbb_min"] = ak.min(mbb, axis=1)
        self.events["mbb_max"] = ak.max(mbb, axis=1)
        self.events["deltaRbb_avg"] = ak.mean(deltaR_unique, axis=1)
        self.events["ptbb_closest"] = ptbb[:,0]
        self.events["htbb_closest"] = htbb[:,0]

        # Define labels for btagged jets at different working points
        for wp, val in self.params.btagging.working_point[self._year]["btagging_WP"].items():
            self.events["JetGood"] = ak.with_field(
                self.events.JetGood,
                ak.values_astype(self.events.JetGood[self.params.btagging.working_point[self._year]["btagging_algorithm"]] > val, int),
                f"btag_{wp}"
            )

    def do_parton_matching(self) -> ak.Array:

        parton_analysis = jet_parton_matching.do_genmatching(self.events, dr_max_postfsr=self.dr_max_postfsr)
        [ higgs, top, antitop, isr,
          quarks_initial, quarks_lastcopy,
          part_from_Wlep, part_from_Whad,
          W_from_top_islep, W_from_antitop_islep,
          b_from_top, b_from_antitop, b_from_higgs ] = parton_analysis
        # Calling our general object_matching function.
        # The output is an awkward array with the shape of the second argument and None where there is no matching.
        # So, calling like this, we will get out an array of matched_quarks with the dimension of the JetGood.
        (matched_quarks, matched_jets, deltaR_matched,
         idx_quarks, idx_jets, _ ) = object_matching(
            quarks_lastcopy, self.events.JetGood, dr_min=self.dr_max, return_indices=True
        )
        #Saving stuff
        self.events["JetGoodMatched"] = ak.with_field(
            matched_jets, deltaR_matched, "dRMatchedJet"
        )
        self.events["JetGoodMatched"] = ak.with_field(
            self.events.JetGoodMatched, matched_quarks.provenance, "provenance")
        
        self.events["PartonInitial"] = quarks_initial
        self.events["PartonLastCopy"] = quarks_lastcopy
        # Saving the matched partons only
        self.events["PartonLastCopyMatched"] = matched_quarks
        # partons are aligned, so I can use the indices of quarks_lastcopy to get the initial
        self.events["PartonInitialMatched"] = quarks_initial[idx_quarks]

        self.events["LeptonGenLevel"] = part_from_Wlep
        self.events["HiggsGen"] = higgs
        
        self.events["TopGen"] = top
        self.events["AntiTopGen"] = antitop
        self.events["TopGen_islep"] = W_from_top_islep
        self.events["AntiTopGen_islep"] = W_from_antitop_islep
        self.events["ISR"] = isr
        

    def count_partons(self):
        self.events["nParton"] = ak.num(self.events.PartonInitial, axis=1)
        self.events["nPartonMatched"] = ak.count(
            self.events.JetGoodMatched.pt, axis=1
        )  # use count since we have None

    def process_extra_after_presel(self, variation) -> ak.Array:
        if self._isMC & (self._sample in ["ttHTobb", "ttHTobb_ttToSemiLep", "ttHTobb_EFT"]): #"TTbbSemiLeptonic", "TTToSemiLeptonic"]):
            self.do_parton_matching()
            self.count_partons()
