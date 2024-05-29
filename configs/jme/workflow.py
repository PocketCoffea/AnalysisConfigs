import awkward as ak

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.hist_manager import Axis

from pocket_coffea.lib.deltaR_matching import object_matching, deltaR_matching_nonunique
from custom_cut_functions import *
from custom_functions import *

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
        self.closure_function_dict = self.workflow_options["closure_function_dict"]

    def add_neutrinos_to_genjets(self, genjets, neutrinos):
        neutrinos_matched = deltaR_matching_nonunique(genjets, neutrinos, 0.4)

        # sum the 4-vecs of all the matched neutrinos an save just the 4-vec of the sum
        neutrinos_matched_sum_px = ak.sum(neutrinos_matched.px, axis=-1)
        neutrinos_matched_sum_py = ak.sum(neutrinos_matched.py, axis=-1)
        neutrinos_matched_sum_pz = ak.sum(neutrinos_matched.pz, axis=-1)
        neutrinos_matched_sum_e = ak.sum(neutrinos_matched.energy, axis=-1)

        # compute pt, eta, phi, mass of the sum
        neutrinos_matched_sum_pt = ak.nan_to_num(
            np.sqrt(neutrinos_matched_sum_px**2 + neutrinos_matched_sum_py**2), nan=0
        )
        neutrinos_matched_sum_eta = ak.nan_to_num(
            np.arctanh(
                neutrinos_matched_sum_pz
                / np.sqrt(neutrinos_matched_sum_pt**2 + neutrinos_matched_sum_pz**2)
            ),
            nan=0,
        )
        neutrinos_matched_sum_phi = ak.nan_to_num(
            np.arctan2(neutrinos_matched_sum_py, neutrinos_matched_sum_px), nan=0
        )
        neutrinos_matched_sum_mass = ak.nan_to_num(
            np.sqrt(
                neutrinos_matched_sum_e**2
                - neutrinos_matched_sum_pt**2
                - neutrinos_matched_sum_pz**2
            ),
            nan=0,
        )

        # create the 4-vec of the sum
        neutrinos_matched_sum = ak.zip(
            {
                "pt": neutrinos_matched_sum_pt,
                "eta": neutrinos_matched_sum_eta,
                "phi": neutrinos_matched_sum_phi,
                "mass": neutrinos_matched_sum_mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )

        # recompute the matched jets quadrivector summing the 4-vecs of the genjets and the gen neutrinos
        genjets_with_neutrinos = genjets + neutrinos_matched_sum

        # print(genjets_with_neutrinos.pt[neutrinos_matched_sum.pt > 0])
        # print(neutrinos_matched_sum.pt[neutrinos_matched_sum.pt > 0])
        # print(genjets.pt[neutrinos_matched_sum.pt > 0])

        genjets_with_neutrinos = ak.with_field(
            genjets,
            genjets_with_neutrinos.pt,
            "pt",
        )
        genjets_with_neutrinos = ak.with_field(
            genjets_with_neutrinos,
            genjets_with_neutrinos.eta,
            "eta",
        )
        genjets_with_neutrinos = ak.with_field(
            genjets_with_neutrinos,
            genjets_with_neutrinos.phi,
            "phi",
        )
        genjets_with_neutrinos = ak.with_field(
            genjets_with_neutrinos,
            genjets_with_neutrinos.mass,
            "mass",
        )

        return genjets_with_neutrinos

    def closure_function(self, eta, pt):
        corr = ak.ones_like(eta)

        function_string = self.closure_function_dict["function_string"]
        corrections_eta_bins = self.closure_function_dict["corrections_eta_bins"]
        num_params = self.closure_function_dict["num_params"]
        jet_pt = self.closure_function_dict["jet_pt"]
        params = self.closure_function_dict["params"]

        function = string_to_pol_function(function_string)

        for i in range(len(corrections_eta_bins[0])):
            mask_eta = (corrections_eta_bins[0][i] <= eta) & (
                eta < corrections_eta_bins[1][i]
            )
            # print(mask_eta)
            mask_pt = (jet_pt[0][i] <= pt) & (pt < jet_pt[1][i])
            # print(mask_pt)
            corr = ak.where(
                mask_eta,
                ak.where(
                    mask_pt,
                    function(pt, *params[i]),
                    ak.where(
                        pt < jet_pt[0][i],
                        function(jet_pt[0][i], *params[i]),
                        function(jet_pt[1][i], *params[i]),
                    ),
                ),
                corr,
            )
            # print(corr)
        return corr

    def apply_object_preselection(self, variation):

        if self._isMC:

            neutrinos = self.events["GenPart"][
                (abs(self.events.GenPart.pdgId) == 12)
                | (abs(self.events.GenPart.pdgId) == 14)
                | (abs(self.events.GenPart.pdgId) == 16)
            ]
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

                # add the energy of the gen neutrinos
                # match genjet with all the gen neutrinos with DeltaR<0.4
                # if a neutrino is matched with more than one genjet, choose the closest one
                # then sum the 4-vecs of all the matched neutrinos an save just the 4-vec of the sum
                # then recompute the matched jets quadrivector summing the 4-vecs of the genjets and the gen neutrinos
                # and then do the a new matching with the reco jets
                self.events["GenJetNeutrino"] = self.add_neutrinos_to_genjets(
                    self.events["GenJet"][mask_flav], neutrinos
                )
                (
                    self.events["GenJetNeutrinoMatched"],
                    self.events["JetNeutrinoMatched"],
                    deltaR_Neutrino_matched,
                ) = object_matching(
                    self.events["GenJetNeutrino"], self.events["Jet"], 0.2  # 0.4
                )

            else:
                mask_pt = (
                    self.events["Jet"].pt * (1 - self.events["Jet"].rawFactor)
                    > 0  # < 12
                )  # HERE #cut on pt_raw>8 inside the file
                (
                    self.events["GenJetMatched"],
                    self.events["JetMatched"],
                    deltaR_matched,
                ) = object_matching(
                    self.events["GenJet"], self.events["Jet"][mask_pt], 0.2  # 0.4
                )
                self.events["GenJetNeutrino"] = self.add_neutrinos_to_genjets(
                    self.events["GenJet"], neutrinos
                )
                (
                    self.events["GenJetNeutrinoMatched"],
                    self.events["JetNeutrinoMatched"],
                    deltaR_Neutrino_matched,
                ) = object_matching(
                    self.events["GenJetNeutrino"], self.events["Jet"], 0.2  # 0.4
                )

            # print("GenJet", self.events["GenJet"].pt[-7])
            # print("GenJetNeutrino", self.events["GenJetNeutrino"].pt[-7])
            # print(neutrinos.pt[-7])

            # # breakpoint()
            # print("GenJetNeutrino", len(self.events["GenJetNeutrino"].pt))
            # print("GenJet", len(self.events["GenJet"].pt))
            # print("GenJetNeutrinoMatched", len(self.events["GenJetNeutrinoMatched"].pt))
            # print("GenJetMatched", len(self.events["GenJetMatched"].pt))

            # remove the None values
            self.events["GenJetMatched"] = self.events.GenJetMatched[
                ~ak.is_none(self.events.GenJetMatched, axis=1)
            ]
            self.events["JetMatched"] = self.events.JetMatched[
                ~ak.is_none(self.events.JetMatched, axis=1)
            ]
            self.events["GenJetNeutrinoMatched"] = self.events.GenJetNeutrinoMatched[
                ~ak.is_none(self.events.GenJetNeutrinoMatched, axis=1)
            ]
            self.events["JetNeutrinoMatched"] = self.events.JetNeutrinoMatched[
                ~ak.is_none(self.events.JetNeutrinoMatched, axis=1)
            ]
            # print("GenJetNeutrinoMatched", len(self.events["GenJetNeutrinoMatched"].pt))
            # print("GenJetMatched", len(self.events["GenJetMatched"].pt))

            deltaR_matched = deltaR_matched[~ak.is_none(deltaR_matched, axis=1)]
            deltaR_Neutrino_matched = deltaR_Neutrino_matched[
                ~ak.is_none(deltaR_Neutrino_matched, axis=1)
            ]

            self.events[f"MatchedJets"] = ak.with_field(
                self.events.GenJetMatched,
                self.events.JetMatched.pt / self.events.GenJetMatched.pt,
                "ResponseJEC",
            )
            self.events[f"MatchedJets"] = ak.with_field(
                self.events.MatchedJets,
                self.events.MatchedJets.ResponseJEC
                * (1 - self.events.JetMatched.rawFactor),
                "ResponseRaw",
            )
            self.events[f"MatchedJets"] = ak.with_field(
                self.events.MatchedJets,
                self.events.JetMatched.eta,
                "RecoEta",
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
            # self.events[f"MatchedJets"] = ak.with_field(
            #     self.events.MatchedJets,
            #     self.events.JetMatched.eta - self.events.MatchedJets.eta,
            #     "DeltaEta",
            # )
            # self.events[f"MatchedJets"] = ak.with_field(
            #     self.events.MatchedJets,
            #     self.events.JetMatched.phi - self.events.MatchedJets.phi,
            #     "DeltaPhi",
            # )
            # self.events[f"MatchedJets"] = ak.with_field(
            #     self.events.MatchedJets,
            #     deltaR_matched,
            #     "DeltaR",
            # )

            # self.events[f"MatchedJets"] = ak.with_field(
            #     self.events.MatchedJets,
            #     self.events.JetMatched.eta / self.events.MatchedJets.eta,
            #     "EtaRecoGen",
            # )

            if int(os.environ.get("PNET", 0)) == 1:
                # PNetReg
                self.events[f"MatchedJets"] = ak.with_field(
                    self.events.MatchedJets,
                    self.events.MatchedJets.ResponseRaw
                    * self.events.JetMatched.PNetRegPtRawCorr,
                    "ResponsePNetReg",
                )
                self.events[f"MatchedJets"] = ak.with_field(
                    self.events.MatchedJets,
                    self.events.MatchedJets.JetPtRaw
                    * self.events.JetMatched.PNetRegPtRawCorr,
                    "JetPtPNetReg",
                )
                self.events[f"MatchedJets"] = ak.with_field(
                    self.events.MatchedJets,
                    ak.where(
                        abs(self.events.MatchedJets.RecoEta) > 4.7,
                        self.events.MatchedJets.ResponseRaw,
                        self.events.MatchedJets.ResponsePNetReg,
                    ),
                    "ResponsePNetReg",
                )
                self.events[f"MatchedJets"] = ak.with_field(
                    self.events.MatchedJets,
                    ak.where(
                        abs(self.events.MatchedJets.RecoEta) > 4.7,
                        self.events.MatchedJets.JetPtRaw,
                        self.events.MatchedJets.JetPtPNetReg,
                    ),
                    "JetPtPNetReg",
                )
                self.events[f"MatchedJets"] = ak.with_field(
                    self.events.MatchedJets,
                    ak.where(
                        self.events.MatchedJets.JetPtRaw < 15,
                        ak.where(self.events.MatchedJets.ResponseRaw < 1, 0, 2),
                        self.events.MatchedJets.ResponsePNetReg,
                    ),
                    "ResponsePNetReg",
                )
                self.events[f"MatchedJets"] = ak.with_field(
                    self.events.MatchedJets,
                    ak.where(
                        self.events.MatchedJets.JetPtRaw < 15,
                        self.events.MatchedJets.JetPtRaw,
                        self.events.MatchedJets.JetPtPNetReg,
                    ),
                    "JetPtPNetReg",
                )

                # PNetRegNeutrino
                self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                    self.events.GenJetNeutrinoMatched,
                    self.events.JetNeutrinoMatched.eta,
                    "RecoEta",
                )
                self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                    self.events.MatchedJetsNeutrino,
                    self.events.MatchedJetsNeutrino.pt
                    * (1 - self.events.JetNeutrinoMatched.rawFactor)
                    * self.events.JetNeutrinoMatched.PNetRegPtRawCorr
                    * self.events.JetNeutrinoMatched.PNetRegPtRawCorrNeutrino,
                    "JetPtPNetRegNeutrino",
                )

                self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                    self.events.MatchedJetsNeutrino,
                    self.events.MatchedJetsNeutrino.JetPtPNetRegNeutrino
                    / self.events.GenJetNeutrinoMatched.pt,
                    "ResponsePNetRegNeutrino",
                )

                # when regression is not valid
                self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                    self.events.MatchedJetsNeutrino,
                    ak.where(
                        abs(self.events.MatchedJetsNeutrino.RecoEta) > 4.7,
                        self.events.JetNeutrinoMatched.pt
                        / self.events.GenJetMatched.pt
                        * (1 - self.events.JetNeutrinoMatched.rawFactor),
                        self.events.MatchedJetsNeutrino.ResponsePNetRegNeutrino,
                    ),
                    "ResponsePNetRegNeutrino",
                )
                self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                    self.events.MatchedJetsNeutrino,
                    ak.where(
                        abs(self.events.MatchedJetsNeutrino.RecoEta) > 4.7,
                        self.events.JetNeutrinoMatched.pt
                        * (1 - self.events.JetNeutrinoMatched.rawFactor),
                        self.events.MatchedJetsNeutrino.JetPtPNetRegNeutrino,
                    ),
                    "JetPtPNetRegNeutrino",
                )
                self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                    self.events.MatchedJetsNeutrino,
                    ak.where(
                        self.events.JetNeutrinoMatched.pt
                        * (1 - self.events.JetNeutrinoMatched.rawFactor)
                        < 15,
                        ak.where(
                            self.events.JetNeutrinoMatched.pt
                            / self.events.GenJetMatched.pt
                            * (1 - self.events.JetNeutrinoMatched.rawFactor)
                            < 1,
                            0,
                            2,
                        ),
                        self.events.MatchedJetsNeutrino.ResponsePNetRegNeutrino,
                    ),
                    "ResponsePNetRegNeutrino",
                )
                self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                    self.events.MatchedJetsNeutrino,
                    ak.where(
                        self.events.JetNeutrinoMatched.pt
                        * (1 - self.events.JetNeutrinoMatched.rawFactor)
                        < 15,
                        self.events.JetNeutrinoMatched.pt
                        * (1 - self.events.JetNeutrinoMatched.rawFactor),
                        self.events.MatchedJetsNeutrino.JetPtPNetRegNeutrino,
                    ),
                    "JetPtPNetRegNeutrino",
                )

                # TODO: apply the closure function even when the regression is not valid
                if self.closure_function_dict:
                    closure_factor = self.closure_function(
                        self.events.MatchedJetsNeutrino.RecoEta,
                        self.events.MatchedJetsNeutrino.JetPtPNetRegNeutrino,
                    )
                    self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                        self.events.MatchedJetsNeutrino,
                        self.events.MatchedJetsNeutrino.JetPtPNetRegNeutrino
                        * closure_factor,
                        "JetPtPNetRegNeutrino",
                    )
                    self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                        self.events.MatchedJetsNeutrino,
                        self.events.MatchedJetsNeutrino.ResponsePNetRegNeutrino
                        * closure_factor,
                        "ResponsePNetRegNeutrino",
                    )

                # jet pet when <15?
                # # set the response to 0 if the PNetReg is not valid
                # #Raw
                # self.events[f"MatchedJets"] = ak.with_field(
                #     self.events.MatchedJets,
                #     ak.where(
                #         self.events.MatchedJets.ResponsePNetReg > 0,
                #         self.events.MatchedJets.ResponseRaw,
                #         0,
                #     ),
                #     "ResponseRaw",
                # )
                # # set the jet pt to 0 if the PNetReg is not valid
                # self.events[f"MatchedJets"] = ak.with_field(
                #     self.events.MatchedJets,
                #     ak.where(
                #         self.events.MatchedJets.JetPtPNetReg > 0,
                #         self.events.MatchedJets.JetPtRaw,
                #         0,
                #     ),
                #     "JetPtRaw",
                # )

                # #JEC
                # self.events[f"MatchedJets"] = ak.with_field(
                #     self.events.MatchedJets,
                #     ak.where(
                #         self.events.MatchedJets.ResponsePNetReg > 0,
                #         self.events.MatchedJets.ResponseJEC,
                #         0,
                #     ),
                #     "ResponseJEC",
                # )
                # self.events[f"MatchedJets"] = ak.with_field(
                #     self.events.MatchedJets,
                #     ak.where(
                #         self.events.MatchedJets.JetPtPNetReg > 0,
                #         self.events.MatchedJets.JetPtJEC,
                #         0,
                #     ),
                #     "JetPtJEC",
                # )

                # self.events[f"MatchedJets"] = ak.with_field(
                #     self.events.MatchedJets,
                #     self.events.MatchedJets.ResponsePNetReg,
                #     "ResponsePNetRegNeutrino",
                # )
                # self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                #     self.events.MatchedJetsNeutrino,
                #     self.events.MatchedJetsNeutrino.ResponsePNetRegNeutrino,
                #     "ResponsePNetReg",
                # )
                # self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                #     self.events.MatchedJetsNeutrino,
                #     self.events.MatchedJetsNeutrino.ResponsePNetRegNeutrino,
                #     "ResponseRaw",
                # )
                # self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                #     self.events.MatchedJetsNeutrino,
                #     self.events.MatchedJetsNeutrino.ResponsePNetRegNeutrino,
                #     "ResponseJEC",
                # )

                # # reshape MatchedJetsNeutrino to be the same shape as MatchedJets
                # self.events[f"MatchedJetsNeutrino_reshape"] = ak.broadcast_arrays(
                #     self.events.MatchedJetsNeutrino, self.events.MatchedJets
                # )[0]

            # gen jet flavour splitting
            if flavour != "inclusive":
                for flav, parton_flavs in flav_dict.items():
                    self.events[f"MatchedJets_{flav}"] = genjet_selection_flavsplit(
                        self.events, "MatchedJets", parton_flavs
                    )
                    self.events[f"MatchedJetsNeutrino_{flav}"] = (
                        genjet_selection_flavsplit(
                            self.events, "MatchedJetsNeutrino", parton_flavs
                        )
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
