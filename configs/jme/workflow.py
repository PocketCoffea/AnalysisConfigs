import awkward as ak

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.hist_manager import Axis
from pocket_coffea.lib.objects import (
    jet_correction,
    lepton_selection,
    jet_selection,
    btagging,
    get_dilepton,
)
from pocket_coffea.lib.deltaR_matching import object_matching
from custom_cut_functions import *

from params.binning import *

from time import sleep
import os

# flav_dict = {
#     "b": 5,
#     "c": 4,
#     "uds": 1,
#     "g": 21,
# }
flav_dict = {
    "b": [5],
    "c": [4],
    "uds": [1, 2, 3],
    "g": [21],
} if int(os.environ.get("FLAVSPLIT", 0)) == 1 else {}


class QCDBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    def apply_object_preselection(self, variation):
        self.events["JetGood"], self.jetGoodMask = jet_selection_nopu(
            self.events, "Jet", self.params  # , "LeptonGood"
        )

        # self.events["JetGood"], self.jetGoodMask = self.events["Jet"], ak.ones_like(
        #     self.events["Jet"].pt, dtype=bool
        # )
        if self._isMC:
            self.events["GenJetGood"], self.genjetGoodMask = jet_selection_nopu(
                self.events, "GenJet", self.params  # , "LeptonGood"
            )

            # Ngenjet = ak.num(self.events.GenJet)
            # matching_with_none = ak.mask(self.events.Jet.genJetIdx, (self.events.Jet.genJetIdx < Ngenjet)&(self.events.Jet.genJetIdx!=-1))
            # self.events["GenJetsOrder"] = self.events.GenJet[matching_with_none]
            # not_none = ~ak.is_none(self.events.GenJetsOrder, axis=1)
            # self.events["JetMatched"] = ak.mask(self.events.Jet, not_none)

            (
                self.events["GenJetMatched"],
                self.events["JetMatched"],
                deltaR_matched,
            ) = object_matching(self.events["GenJetGood"], self.events["JetGood"], 0.2)

            self.events["MatchedJets_inclusive"] = ak.with_field(
                self.events.GenJetMatched,
                self.events.JetMatched.pt / self.events.GenJetMatched.pt,
                "Response",
            )
            if int(os.environ.get("PNET", 0)) == 1:
                self.events["MatchedJets_inclusive"] = ak.with_field(
                    self.events.MatchedJets_inclusive,
                    self.events.JetMatched.pt
                    * self.events.JetMatched.PNetRegPtRawCorr
                    / self.events.GenJetMatched.pt,
                    "ResponsePNetReg",
                )

            self.events["MatchedJets_inclusive"] = self.events.MatchedJets_inclusive[
                ~ak.is_none(self.events.MatchedJets_inclusive, axis=1)
            ]

            # gen jet flavour splitting
            for flav, parton_flavs in flav_dict.items():
                self.events[f"MatchedJets_{flav}"] = genjet_selection_flavsplit(
                    self.events, "MatchedJets_inclusive", parton_flavs
                )

            # self.events["MatchedJets"] = ak.with_field(
            #     self.events["MatchedJets"],
            #     abs(self.events["MatchedJets"].eta),
            #     "AbsEta",
            # )

            # for i in range(len(eta_bins) - 1):
            #     for j in range(len(pt_bins) - 1):
            #         eta_min = eta_bins[i]
            #         eta_max = eta_bins[i + 1]
            #         pt_min = pt_bins[j]
            #         pt_max = pt_bins[j + 1]
            #         name = f"MatchedJets_eta{eta_min}-{eta_max}_pt{pt_min}-{pt_max}"
            #         mask_eta = ((self.events.MatchedJets.eta) > eta_min) & (
            #             (self.events.MatchedJets.eta) < eta_max
            #         )  # mask for jets in eta bin
            #         mask_pt = (self.events.MatchedJets.pt > pt_min) & (
            #             self.events.MatchedJets.pt < pt_max
            #         )
            #         mask = mask_eta & mask_pt
            #         self.events[name] = self.events.MatchedJets[mask]

            if int(os.environ.get("CARTESIAN", 0)) == 1:
                return

            for j in range(len(pt_bins) - 1):
                # read eta_min for the environment variable ETA_MIN
                eta_min = float(os.environ.get("ETA_MIN", -999.0))
                eta_max = float(os.environ.get("ETA_MAX", -999.0))
                pt_min = pt_bins[j]
                pt_max = pt_bins[j + 1]
                mask_pt = (self.events.MatchedJets_inclusive.pt > pt_min) & (
                    self.events.MatchedJets_inclusive.pt < pt_max
                )
                if eta_min != -999.0 and eta_max != -999.0:
                    name = f"MatchedJets_inclusive_eta{eta_min}to{eta_max}_pt{pt_min}to{pt_max}"
                    mask_eta = ((self.events.MatchedJets_inclusive.eta) > eta_min) & (
                        (self.events.MatchedJets_inclusive.eta) < eta_max
                    )
                    mask = mask_eta & mask_pt
                    mask = mask[~ak.is_none(mask, axis=1)]

                else:
                    name = f"MatchedJets_inclusive_pt{pt_min}to{pt_max}"
                    mask = mask_pt
                    mask = ak.mask(mask, mask)
                self.events[name] = self.events.MatchedJets_inclusive[mask]

            # for j in range(len(pt_bins) - 1):
            #     pt_min = pt_bins[j]
            #     pt_max = pt_bins[j + 1]
            #     name = f"MatchedJets_pt{pt_min}to{pt_max}"
            #     mask = (self.events.MatchedJets.pt > pt_min) & (
            #         self.events.MatchedJets.pt < pt_max
            #     )
            #     # put None where mask is False
            #     mask = ak.mask(mask, mask)

            #     # mask = ak.where(ak.is_none(mask, axis=1), False, mask)
            #     # mask=mask[~ak.is_none(mask, axis=1)]
            #     self.events[name] = self.events.MatchedJets[mask]

    def count_objects(self, variation):
        self.events["nJetGood"] = ak.num(self.events.JetGood)
        if self._isMC:
            self.events["nGenJetGood"] = ak.num(self.events.GenJetGood)

    # # Function that defines common variables employed in analyses and save them as attributes of `events`
    # def define_common_variables_after_presel(self, variation):
    #     if self._isMC:
    #         # self.events["JetMatched"] = ak.with_field(self.events.JetMatched, self.events.JetMatched.pt/self.events.GenJetsOrder.pt, "Response_old")
    #         # self.events["JetMatched"] = ak.with_field(self.events.JetMatched, self.events.JetMatched.delta_r(self.events.GenJetsOrder), "DeltaR_old")

    #         self.events["MatchedJets"] = ak.with_field(
    #             self.events.MatchedJets,
    #             self.events.JetMatched.pt / self.events.MatchedJets.pt,
    #             "Response",
    #         )

    #         self.events["MatchedJets"] = ak.with_field(
    #             self.events["MatchedJets"],
    #             abs(self.events["MatchedJets"].eta),
    #             "AbsEta",
    #         )

    #         # questo sotto non va bene perchè viene calcolata indipendentemente per ogni chunk
    #         # print(f"computing median for JetGood")
    #         # print(self.events["JetGood"].pt)
    #         # print(ak.flatten(self.events["JetGood"].pt))
    #         # print(np.median(ak.to_numpy(ak.flatten(self.events["JetGood"].pt))))
    #         # print("\n")

    #         # define multiple objects dividing jets in eta bins
    #         for i in range(len(eta_bins) - 1):
    #             for j in range(len(pt_bins) - 1):
    #                 eta_min = eta_bins[i]
    #                 eta_max = eta_bins[i + 1]
    #                 pt_min = pt_bins[j]
    #                 pt_max = pt_bins[j + 1]
    #                 name = f"MatchedJets_eta{eta_min}-{eta_max}_pt{pt_min}-{pt_max}"
    #                 mask_eta = (abs(self.events.MatchedJets.eta) > eta_min) & (
    #                     abs(self.events.MatchedJets.eta) < eta_max
    #                 )  # mask for jets in eta bin
    #                 mask_pt = (self.events.MatchedJets.pt > pt_min) & (
    #                     self.events.MatchedJets.pt < pt_max
    #                 )
    #                 mask = mask_eta & mask_pt
    #                 self.events[name] = self.events.MatchedJets[mask]

    #                 # questo sotto non va bene perchè viene calcolata indipendentemente per ogni chunk
    #                 # print(f"computing median for {name}")
    #                 # print(self.events[name].Response)
    #                 # print(ak.flatten(self.events[name].Response))
    #                 # print(np.median(ak.to_numpy(ak.flatten(self.events[name].Response))))

    #                 # compute the median of the Response distribution
    #                 # self.events[name] = ak.with_field(
    #                 #     self.events[name],
    #                 #     ak.median(self.events[name].Response),
    #                 #     "MedianResponse",
    #                 # )
