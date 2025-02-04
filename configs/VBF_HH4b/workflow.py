import awkward as ak
import sys
import numpy as np

from configs.HH4b_common.custom_cut_functions_common import jet_selection_nopu
from configs.HH4b_common.workflow_common import HH4bCommonProcessor
from utils.vbf_matching import get_jets_no_higgs
from utils.basic_functions import add_fields


class VBFHH4bProcessor(HH4bCommonProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.vbf_parton_matching = self.workflow_options["vbf_parton_matching"]

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation=variation)

        self.events["JetVBF_matching"] = self.events.Jet
        self.events["JetVBF_matching"] = jet_selection_nopu(
            self.events, "JetVBF_matching", self.params
        )

        self.events["JetGoodVBF"] = self.events.Jet
        self.events["JetGoodVBF"] = jet_selection_nopu(
            self.events, "JetGoodVBF", self.params
        )

        self.events["JetVBF_generalSelection"] = self.events.Jet
        self.events["JetVBF_generalSelection"] = jet_selection_nopu(
            self.events, "JetVBF_generalSelection", self.params
        )

    def count_objects(self, variation):
        super().count_objects(variation=variation)
        self.events["nJetGoodVBF"] = ak.num(self.events.JetGoodVBF, axis=1)
        self.events["nJetVBF_generalSelection"] = ak.num(
            self.events.JetVBF_generalSelection, axis=1
        )

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        super().process_extra_after_presel(variation=variation)

        if self._isMC:
            jet_offsets = np.concatenate(
                [
                    [0],
                    np.cumsum(
                        ak.to_numpy(ak.num(self.events.Jet, axis=1), allow_missing=True)
                    ),
                ]
            )
            local_index_all = ak.local_index(self.events.Jet, axis=1)
            jets_index_all = ak.to_numpy(
                ak.flatten(local_index_all + jet_offsets[:-1]), allow_missing=True
            )
            jets_from_higgs_idx = ak.to_numpy(
                ak.flatten(self.matched_jet_higgs_idx_not_none + jet_offsets[:-1]),
                allow_missing=False,
            )
            jets_no_higgs_idx = get_jets_no_higgs(jets_index_all, jets_from_higgs_idx)
            jets_no_higgs_idx_unflat = (
                ak.unflatten(jets_no_higgs_idx, ak.num(self.events.Jet, axis=1))
                - jet_offsets[:-1]
            )
            self.events["JetVBFNotFromHiggs"] = self.events.Jet[
                jets_no_higgs_idx_unflat >= 0
            ]
            # apply selection to the jets not from Higgs
            self.events["JetVBFNotFromHiggs"] = jet_selection_nopu(
                self.events, "JetVBFNotFromHiggs", self.params
            )

            # order in pt
            self.events["JetVBFNotFromHiggs"] = self.events.JetVBFNotFromHiggs[
                ak.argsort(self.events.JetVBFNotFromHiggs.pt, axis=1, ascending=False)
            ]

            self.events["HH"] = add_fields(
                self.events.HiggsLeading + self.events.HiggsSubLeading
            )

            if self.vbf_parton_matching:
                self.do_vbf_parton_matching(which_bquark=self.which_bquark)

                self.events["nJetVBF_matched"] = ak.num(
                    self.events.JetVBF_matched, axis=1
                )

                # Create new variable delta eta and invariant mass of the jets
                JetVBF_matched_padded = ak.pad_none(
                    self.events.JetVBF_matched, 2
                )  # Adds none jets to events that have less than 2 jets

                self.events["deltaEta_matched"] = abs(
                    JetVBF_matched_padded.eta[:, 0] - JetVBF_matched_padded.eta[:, 1]
                )

                self.events["jj_mass_matched"] = (
                    JetVBF_matched_padded[:, 0] + JetVBF_matched_padded[:, 1]
                ).mass

                # This product will give only -1 or 1 values, as it's needed to see if the two jets are in the same side or not
                self.events["etaProduct"] = (
                    JetVBF_matched_padded.eta[:, 0] * JetVBF_matched_padded.eta[:, 1]
                ) / abs(
                    JetVBF_matched_padded.eta[:, 0] * JetVBF_matched_padded.eta[:, 1]
                )

            # choose vbf jets as the two jets with the highest pt that are not from higgs decay
            self.events["JetVBFLeadingPtNotFromHiggs"] = self.events.JetVBFNotFromHiggs[
                :, :2
            ]

            # choose higgs jets as the two jets with the highest mjj that are not from higgs decay
            jet_combinations = ak.combinations(self.events.JetVBFNotFromHiggs, 2)
            jet_combinations_mass = (jet_combinations["0"] + jet_combinations["1"]).mass
            jet_combinations_mass_max_idx = ak.to_numpy(
                ak.argsort(jet_combinations_mass, axis=1, ascending=False)[:, 0]
            )

            jets_max_mass = jet_combinations[
                ak.local_index(jet_combinations, axis=0), jet_combinations_mass_max_idx
            ]
            vbf_jets_max_mass_0 = ak.unflatten(
                self.events.Jet[
                    ak.local_index(self.events.Jet, axis=0),
                    ak.to_numpy(jets_max_mass["0"].index),
                ],
                1,
            )
            vbf_jets_max_mass_1 = ak.unflatten(
                self.events.Jet[
                    ak.local_index(self.events.Jet, axis=0),
                    ak.to_numpy(jets_max_mass["1"].index),
                ],
                1,
            )

            vbf_jet_leading_mjj = ak.with_name(
                ak.concatenate([vbf_jets_max_mass_0, vbf_jets_max_mass_1], axis=1),
                name="PtEtaPhiMCandidate",
            )

            vbf_jet_leading_mjj_fields_dict = {
                field: getattr(vbf_jet_leading_mjj, field)
                for field in vbf_jet_leading_mjj.fields
                if ("muon" not in field and "electron" not in field)
            }
            self.events["JetVBFLeadingMjjNotFromHiggs"] = add_fields(
                vbf_jet_leading_mjj, vbf_jet_leading_mjj_fields_dict
            )

            self.events["JetVBFLeadingPtNotFromHiggs_deltaEta"] = abs(
                self.events.JetVBFLeadingPtNotFromHiggs.eta[:, 0]
                - self.events.JetVBFLeadingPtNotFromHiggs.eta[:, 1]
            )

            self.events["JetVBFLeadingMjjNotFromHiggs_deltaEta"] = abs(
                self.events.JetVBFLeadingMjjNotFromHiggs.eta[:, 0]
                - self.events.JetVBFLeadingMjjNotFromHiggs.eta[:, 1]
            )

            self.events["JetVBFLeadingPtNotFromHiggs_jjMass"] = (
                self.events.JetVBFLeadingPtNotFromHiggs[:, 0]
                + self.events.JetVBFLeadingPtNotFromHiggs[:, 1]
            ).mass

            self.events["JetVBFLeadingMjjNotFromHiggs_jjMass"] = (
                self.events.JetVBFLeadingMjjNotFromHiggs[:, 0]
                + self.events.JetVBFLeadingMjjNotFromHiggs[:, 1]
            ).mass

            self.events["HH_deltaR"] = self.events.HiggsLeading.delta_r(
                self.events.HiggsSubLeading
            )

            self.events["jj_deltaR"] = self.events.JetVBFLeadingPtNotFromHiggs[
                :, 0
            ].delta_r(self.events.JetVBFLeadingPtNotFromHiggs[:, 1])

            self.events["H1j1_deltaR"] = self.events.HiggsLeading.delta_r(
                self.events.JetVBFLeadingPtNotFromHiggs[:, 0]
            )

            self.events["H1j2_deltaR"] = self.events.HiggsLeading.delta_r(
                self.events.JetVBFLeadingPtNotFromHiggs[:, 1]
            )

            self.events["H2j1_deltaR"] = self.events.HiggsSubLeading.delta_r(
                self.events.JetVBFLeadingPtNotFromHiggs[:, 0]
            )

            self.events["H2j2_deltaR"] = self.events.HiggsSubLeading.delta_r(
                self.events.JetVBFLeadingPtNotFromHiggs[:, 1]
            )

            JetVBFLeadingPtNotFromHiggs_etaAverage = (
                self.events.JetVBFLeadingPtNotFromHiggs.eta[:, 0]
                + self.events.JetVBFLeadingPtNotFromHiggs.eta[:, 1]
            ) / 2

            self.events["HH_centrality"] = np.exp(
                (
                    -(
                        (
                            self.events.HiggsLeading.eta
                            - JetVBFLeadingPtNotFromHiggs_etaAverage
                        )
                        ** 2
                    )
                    - (
                        self.events.HiggsSubLeading.eta
                        - JetVBFLeadingPtNotFromHiggs_etaAverage
                    )
                    ** 2
                )
                / (self.events.JetVBFLeadingPtNotFromHiggs_deltaEta) ** 2
            )
            print("HH_centrality", self.events.HH_centrality)
