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

flav_dict = (
    {
        "b": 5,
        "c": 4,
        "uds": [1, 2, 3],
        "g": 21,
    }
    if int(os.environ.get("FLAVSPLIT", 0)) == 1
    else {}
)
# flav_dict = {
#     "b": 5,
#     # "c": 4,
#     # "uds": 1,
#     # "g": 21,
# }

flav_def = {
    "b": 5,
    "c": 4,
    "u": 1,
    "d": 2,
    "s": 3,
    "uds": [1, 2, 3],
    "g": 21,
    "inclusive": [1, 2, 3, 4, 5, 21],
}

flavour = str(os.environ.get("FLAV", "inclusive"))
# flav_dict = {
#     flavour: flav_def[flavour],
# }


print(f"\n flav_dict: {flav_dict}")
print(f"\n flavour: {flavour}")


class QCDBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    def apply_object_preselection(self, variation):
        # self.events["Jet"], self.jetGoodMask = jet_selection_nopu(
        #     self.events, "Jet", self.params  # , "LeptonGood"
        # )

        # self.events["Jet"] = self.events["Jet"]
        if self._isMC:
            # self.events["GenJet"], self.genjetGoodMask = jet_selection_nopu(
            #     self.events, "GenJet", self.params  # , "LeptonGood"
            # )
            # self.events["GenJet"] = self.events["GenJet"]

            # Idx matching
            # Ngenjet = ak.num(self.events["GenJet"])
            # matched_genjets_idx = ak.mask(
            #     self.events["Jet"].genJetIdx,
            #     (self.events["Jet"].genJetIdx < Ngenjet) & (self.events["Jet"].genJetIdx != -1),
            # )
            # # this array of indices has already the dimension of the Jet collection
            # # in NanoAOD nomatch == -1 --> convert to None with a mask
            # matched_objs_mask = ~ak.is_none(matched_genjets_idx, axis=1)
            # self.events["GenJetMatchedIdx"] = self.events["GenJet"][matched_genjets_idx]
            # self.events["JetMatchedIdx"] = ak.mask(self.events["Jet"], matched_objs_mask)

            # for the flavsplit
            if flavour != "inclusive":
                mask_flav = (
                    self.events["GenJet"].partonFlavour == flav_def[flavour]
                    if type(flav_def[flavour]) == int
                    else ak.any(
                        [
                            self.events["GenJet"].partonFlavour == flav
                            for flav in flav_def[flavour]
                        ],
                        axis=0,
                    )
                )
                # self.events["GenJet_mask"]=self.events.GenJet[mask_flav]
                # self.events["GenJet_mask"] = self.events.GenJet_mask[
                #     ~ak.is_none(self.events.GenJet_mask, axis=1)
                # ]
                # # mask_flav = ak.mask(mask_flav, mask_flav)
                (
                    self.events["GenJetMatched"],
                    self.events["JetMatched"],
                    deltaR_matched,
                ) = object_matching(
                    self.events["GenJet"][mask_flav], self.events["Jet"], 0.2  # 0.4
                )

                # TODO: add the energy of the gen neutrinos
                # match genjet with all the gen neutrinos with DeltaR<0.4
                # then sum the 4-vecs of all the matched neutrinos an save just the 4-vec of the sum
                # then recompute the matched jets quadrivector summing the 4-vecs of the genjets and the gen neutrinos
                # and then do the a new matching with the reco jets


            # elif flav_dict:
            #     for flav, parton_flavs in flav_dict.items():
            #         self.events[f"GenJet_{flav}"] = genjet_selection_flavsplit(
            #             self.events, "GenJet", parton_flavs
            #         )
            #         (
            #             self.events[f"GenJetMatched_{flav}"],
            #             self.events[f"JetMatched_{flav}"],
            #             deltaR_matched,
            #         ) = object_matching(self.events[f"GenJet_{flav}"], self.events["Jet"], 0.2)

            else:
                mask_pt = (
                    self.events["Jet"].pt * (1 - self.events["Jet"].rawFactor) >0 #< 12
                )  # HERE #cut on pt_raw>8 inside the file
                (
                    self.events["GenJetMatched"],
                    self.events["JetMatched"],
                    deltaR_matched,
                ) = object_matching(
                    self.events["GenJet"],
                    self.events["Jet"][mask_pt],
                    0.2  # HERE # 0.2 #0.4
                )

            self.events["GenJetMatched"] = self.events.GenJetMatched[
                ~ak.is_none(self.events.GenJetMatched, axis=1)
            ]
            self.events["JetMatched"] = self.events.JetMatched[
                ~ak.is_none(self.events.JetMatched, axis=1)
            ]

            deltaR_matched = deltaR_matched[~ak.is_none(deltaR_matched, axis=1)]

            # this might be useless
            # if flavour != "inclusive":
            #     mask_flav=self.events["GenJetMatched"].partonFlavour == flav_def[flavour]
            #     mask_flav = ak.mask(mask_flav, mask_flav)
            #     self.events[f"MatchedJets"] = ak.with_field(
            #         self.events.GenJetMatched[mask_flav],
            #         self.events.JetMatched[mask_flav].pt / self.events.GenJetMatched[mask_flav].pt,
            #         "ResponseJEC",
            #     )
            # else:

            self.events[f"MatchedJets"] = ak.with_field(
                self.events.GenJetMatched,
                self.events.JetMatched.pt / self.events.GenJetMatched.pt,
                "ResponseJEC",
            )

            self.events[f"MatchedJets"] = ak.with_field(
                self.events.MatchedJets,
                self.events.JetMatched.pt,
                "JetPtJEC",
            )
            self.events[f"MatchedJets"] = ak.with_field(
                self.events.MatchedJets,
                self.events.JetMatched.pt * (1 - self.events.JetMatched.rawFactor),
                "JetPtRaw",
            )
            self.events[f"MatchedJets"] = ak.with_field(
                self.events.MatchedJets,
                self.events.JetMatched.eta - self.events.MatchedJets.eta,
                "DeltaEta",
            )
            self.events[f"MatchedJets"] = ak.with_field(
                self.events.MatchedJets,
                self.events.JetMatched.phi - self.events.MatchedJets.phi,
                "DeltaPhi",
            )
            self.events[f"MatchedJets"] = ak.with_field(
                self.events.MatchedJets,
                deltaR_matched,
                "DeltaR",
            )

            self.events[f"MatchedJets"] = ak.with_field(
                self.events.MatchedJets,
                self.events.JetMatched.eta / self.events.MatchedJets.eta,
                "EtaRecoGen",
            )

            self.events[f"MatchedJets"] = ak.with_field(
                self.events.MatchedJets,
                self.events.MatchedJets.ResponseJEC
                * (1 - self.events.JetMatched.rawFactor),
                "ResponseRaw",
            )

            if int(os.environ.get("PNET", 0)) == 1:
                self.events[f"MatchedJets"] = ak.with_field(
                    self.events.MatchedJets,
                    self.events.MatchedJets.ResponseRaw
                    * self.events.JetMatched.PNetRegPtRawCorr,
                    "ResponsePNetReg",
                )
                # self.events[f"MatchedJets"] = ak.with_field(
                #     self.events.MatchedJets,
                #     self.events.MatchedJets.ResponseRaw
                #     * self.events.JetMatched.PNetRegPtRawCorrNeutrino,
                #     "ResponsePNetRegNeutrino",
                # )
                # self.events[f"MatchedJets"] = ak.with_field(
                #     self.events.MatchedJets,
                #     self.events.MatchedJets.ResponseRaw
                #     * self.events.JetMatched.PNetRegPtRawCorr
                #     * self.events.JetMatched.PNetRegPtRawCorrNeutrino,
                #     "ResponsePNetRegFull",
                # )
                # self.events[f"JetMatched"] = ak.with_field(
                #     self.events.JetMatched,
                #     self.events.JetMatched.PNetRegPtRawCorr
                #     * self.events.JetMatched.PNetRegPtRawCorrNeutrino,
                #     "PNetRegPtRawCorrFull",
                # )

            # self.events["MatchedJets"] = self.events.MatchedJets[
            #     ~ak.is_none(self.events.MatchedJets, axis=1)
            # ]

            # if flavour != "inclusive":
            #     mask_flav=self.events["MatchedJets"].partonFlavour == flav_def[flavour]
            #     # mask_flav = ak.mask(mask_flav, mask_flav)
            #     self.events[f"MatchedJets_{flavour}"] = self.events["MatchedJets"][
            #         mask_flav
            #     ]

            # gen jet flavour splitting
            if flavour != "inclusive":
                for flav, parton_flavs in flav_dict.items():
                    self.events[f"MatchedJets_{flav}"] = genjet_selection_flavsplit(
                        self.events, "MatchedJets", parton_flavs
                    )

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
                mask_pt = (self.events.MatchedJets.pt > pt_min) & (
                    self.events.MatchedJets.pt < pt_max
                )
                if eta_min != -999.0 and eta_max != -999.0:
                    name = f"MatchedJets_eta{eta_min}to{eta_max}_pt{pt_min}to{pt_max}"
                    mask_eta = ((self.events.MatchedJets.eta) > eta_min) & (
                        (self.events.MatchedJets.eta) < eta_max
                    )
                    mask = mask_eta & mask_pt
                    mask = mask[~ak.is_none(mask, axis=1)]

                else:
                    name = f"MatchedJets_pt{pt_min}to{pt_max}"
                    mask = mask_pt
                    mask = ak.mask(mask, mask)
                self.events[name] = self.events.MatchedJets[mask]

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
        self.events["nJet"] = ak.num(self.events.Jet)
        if self._isMC:
            self.events["nGenJet"] = ak.num(self.events.GenJet)

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
    #         # print(f"computing median for Jet")
    #         # print(self.events["Jet"].pt)
    #         # print(ak.flatten(self.events["Jet"].pt))
    #         # print(np.median(ak.to_numpy(ak.flatten(self.events["Jet"].pt))))
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
