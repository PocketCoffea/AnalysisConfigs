import awkward as ak

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.hist_manager import Axis

from pocket_coffea.lib.deltaR_matching import object_matching, deltaR_matching_nonunique
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



print(f"\n flav_dict: {flav_dict}")
print(f"\n flavour: {flavour}")


class QCDBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    # def add_genpart_to_genjet(self, genjets, genparts, pdgId_list):



    def apply_object_preselection(self, variation):

        if self._isMC:


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

                (
                    self.events["GenJetMatched"],
                    self.events["JetMatched"],
                    deltaR_matched,
                ) = object_matching(
                    self.events["GenJet"][mask_flav], self.events["Jet"], 0.2  # 0.4
                )

                # TODO: add the energy of the gen neutrinos
                # match genjet with all the gen neutrinos with DeltaR<0.4
                # if a neutrino is matched with more than one genjet, choose the closest one
                # then sum the 4-vecs of all the matched neutrinos an save just the 4-vec of the sum
                # then recompute the matched jets quadrivector summing the 4-vecs of the genjets and the gen neutrinos
                # and then do the a new matching with the reco jets

                # gen_jet_with_neutrinos = self.add_genpart_to_genjet(
                #     self.events["GenJet"], self.events["GenPart"], [12, 14, 16])

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


            # gen jet flavour splitting
            if flavour != "inclusive":
                for flav, parton_flavs in flav_dict.items():
                    self.events[f"MatchedJets_{flav}"] = genjet_selection_flavsplit(
                        self.events, "MatchedJets", parton_flavs
                    )


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


    def count_objects(self, variation):
        self.events["nJet"] = ak.num(self.events.Jet)
        if self._isMC:
            self.events["nGenJet"] = ak.num(self.events.GenJet)
