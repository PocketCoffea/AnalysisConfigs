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
        self.mc_truth_corr_pnetreg = self.workflow_options["mc_truth_corr_pnetreg"]
        self.mc_truth_corr_pnetreg_neutrino = self.workflow_options[
            "mc_truth_corr_pnetreg_neutrino"
        ]
        self.mc_truth_corr = self.workflow_options["mc_truth_corr"]
        self.DeltaR_matching = self.workflow_options["DeltaR_matching"]
        self.SetRegResponseToZero = self.workflow_options["SetRegResponseToZero"]
        self.GenJetPtCut = self.workflow_options["GenJetPtCut"]

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

    def get_mc_truth_corr(self, corr_dict, eta, phi, pt, pnetreg=True):
        corr = ak.ones_like(eta)

        function_string = corr_dict["function_string"]
        corrections_eta_bins = corr_dict["corrections_eta_bins"]
        corrections_phi_bins = corr_dict["corrections_phi_bins"]
        num_params = corr_dict["num_params"]
        jet_pt = corr_dict["jet_pt"]
        params = corr_dict["params"]

        if pnetreg:
            corr_function = string_to_pol_function(function_string)
        else:
            corr_function = standard_gaus_function

        pt = ak.values_astype(pt, "float64")

        for i in range(len(corrections_eta_bins[0])):
            mask_bins = (corrections_eta_bins[0][i] <= eta) & (
                eta < corrections_eta_bins[1][i]
            )
            if corrections_phi_bins:
                mask_bins = (
                    mask_bins
                    & (corrections_phi_bins[0][i] <= phi)
                    & (phi < corrections_phi_bins[1][i])
                )
            mask_pt = (jet_pt[0][i] <= pt) & (pt < jet_pt[1][i])
            corr = ak.where(
                mask_bins,
                ak.where(
                    mask_pt,
                    corr_function(pt, *params[i]),
                    ak.where(
                        pt < jet_pt[0][i],
                        corr_function(jet_pt[0][i], *params[i]),
                        corr_function(jet_pt[1][i], *params[i]),
                    ),
                ),
                corr,
            )

        return corr

    def apply_object_preselection(self, variation):

        if self._isMC:
            PV_dz_mask = abs(self.events.PV.z - self.events.GenVtx.z) < 0.2
            self.events = self.events[PV_dz_mask]

            self.events["JetGood"] = ak.with_field(
                self.events.Jet,
                self.events.Jet.pt * (1 - self.events.Jet.rawFactor),
                "ptRaw",
            )

            self.events["JetGood"] = self.events.JetGood[
                ak.argsort(self.events.JetGood.ptRaw, ascending=False)
            ][:, :3]

            physisical_jet_mask = (
                self.events.JetGood.ptRaw * np.cosh(self.events.JetGood.eta)
                < (13.6 * 1000) / 2
            )
            eta_mask = (self.events.JetGood.eta > eta_bins[0]) & (
                self.events.JetGood.eta < eta_bins[-1]
            )

            pt_mask = self.events.JetGood.ptRaw > 8

            self.events["JetGood"] = self.events.JetGood[physisical_jet_mask & eta_mask & pt_mask]

            self.events["GenJetGood"] = self.events.GenJet[
                (self.events.GenJet.pt > pt_bins[0])
                & (self.events.GenJet.pt < pt_bins[-1])
            ]

            neutrinos = self.events["GenPart"][
                (abs(self.events.GenPart.pdgId) == 12)
                | (abs(self.events.GenPart.pdgId) == 14)
                | (abs(self.events.GenPart.pdgId) == 16)
            ]
            # for the flavsplit
            if flavour != "inclusive":
                self.mask_flav = (
                    self.events["GenJetGood"].partonFlavour == flav_def[flavour]
                    if type(flav_def[flavour]) == int
                    else ak.any(
                        [
                            self.events["GenJetGood"].partonFlavour == flav
                            for flav in flav_def[flavour]
                        ],
                        axis=0,
                    )
                )

                # add the energy of the gen neutrinos
                # match genjet with all the gen neutrinos with DeltaR<0.4
                # if a neutrino is matched with more than one genjet, choose the closest one
                # then sum the 4-vecs of all the matched neutrinos an save just the 4-vec of the sum
                # then recompute the matched jets quadrivector summing the 4-vecs of the genjets and the gen neutrinos
                # and then do the a new matching with the reco jets
                self.events["GenJetNeutrino"] = self.add_neutrinos_to_genjets(
                    self.events["GenJetGood"][self.mask_flav], neutrinos
                )
            else:
                if int(os.environ.get("NEUTRINO", 1)) == 1:
                    self.events["GenJetNeutrino"] = self.add_neutrinos_to_genjets(
                        self.events["GenJetGood"], neutrinos
                    )

            if (
                int(os.environ.get("NEUTRINO", 1)) == 1
            ):
                self.events["GenJetNeutrino"] = self.events.GenJetNeutrino[
                    self.events.GenJetNeutrino.pt > self.GenJetPtCut
                ]
            elif(
                 int(os.environ.get("NEUTRINO", 0)) == 0
            ):
                self.events["GenJet"] = self.events.GenJet[
                    self.events.GenJet.pt > self.GenJetPtCut
                ]


    def process_extra_after_presel(self, variation) -> ak.Array:

        if self._isMC:

            if flavour != "inclusive":
                (
                    self.events["GenJetMatched"],
                    self.events["JetMatched"],
                    deltaR_matched,
                ) = object_matching(
                    self.events["GenJetGood"][self.mask_flav],
                    self.events["JetGood"],
                    self.DeltaR_matching,
                )
                (
                    self.events["GenJetNeutrinoMatched"],
                    self.events["JetNeutrinoMatched"],
                    deltaR_Neutrino_matched,
                ) = object_matching(
                    self.events["GenJetNeutrino"],
                    self.events["JetGood"],
                    self.DeltaR_matching,
                )

            else:
                (
                    self.events["GenJetMatched"],
                    self.events["JetMatched"],
                    deltaR_matched,
                ) = object_matching(
                    self.events["GenJetGood"],
                    self.events["JetGood"],
                    self.DeltaR_matching,
                )
                if int(os.environ.get("NEUTRINO", 1)) == 1:
                    (
                        self.events["GenJetNeutrinoMatched"],
                        self.events["JetNeutrinoMatched"],
                        deltaR_Neutrino_matched,
                    ) = object_matching(
                        self.events["GenJetNeutrino"],
                        self.events["JetGood"],
                        self.DeltaR_matching,
                    )

            # remove the None values
            self.events["GenJetMatched"] = self.events.GenJetMatched[
                ~ak.is_none(self.events.GenJetMatched, axis=1)
            ]
            self.events["JetMatched"] = self.events.JetMatched[
                ~ak.is_none(self.events.JetMatched, axis=1)
            ]
            self.events[f"MatchedJets"] = ak.with_field(
                self.events.GenJetMatched,
                self.events.JetMatched.eta,
                "RecoEta",
            )
            self.events[f"MatchedJets"] = ak.with_field(
                self.events.MatchedJets,
                self.events.JetMatched.phi,
                "RecoPhi",
            )

            self.events[f"MatchedJets"] = ak.with_field(
                self.events.MatchedJets,
                self.events.JetMatched.pt * (1 - self.events.JetMatched.rawFactor),
                "JetPtRaw",
            )
            self.events[f"MatchedJets"] = ak.with_field(
                self.events.MatchedJets,
                self.events.MatchedJets.JetPtRaw / self.events.GenJetMatched.pt,
                "ResponseRaw",
            )

            if self.mc_truth_corr:
                mc_truth_corr_factor = self.get_mc_truth_corr(
                    self.mc_truth_corr,
                    self.events.MatchedJets.RecoEta,
                    self.events.MatchedJets.RecoPhi,
                    self.events.MatchedJets.JetPtRaw,
                    pnetreg=False,
                )
                self.events[f"MatchedJets"] = ak.with_field(
                    self.events.MatchedJets,
                    self.events.MatchedJets.JetPtRaw * mc_truth_corr_factor,
                    "JetPtJEC",
                )
                self.events[f"MatchedJets"] = ak.with_field(
                    self.events.MatchedJets,
                    self.events.MatchedJets.ResponseRaw * mc_truth_corr_factor,
                    "ResponseJEC",
                )
                # self.events[f"MatchedJets"] = ak.with_field(
                #     self.events.MatchedJets,
                #     mc_truth_corr_factor,
                #     "MCTruthCorrPNetReg",
                # )
            else:
                self.events[f"MatchedJets"] = ak.with_field(
                    self.events.MatchedJets,
                    self.events.JetMatched.pt,
                    "JetPtJEC",
                )
                self.events[f"MatchedJets"] = ak.with_field(
                    self.events.MatchedJets,
                    self.events.JetMatched.pt / self.events.GenJetMatched.pt,
                    "ResponseJEC",
                )

            if int(os.environ.get("NEUTRINO", 1)) == 1:
                self.events["GenJetNeutrinoMatched"] = (
                    self.events.GenJetNeutrinoMatched[
                        ~ak.is_none(self.events.GenJetNeutrinoMatched, axis=1)
                    ]
                )
                self.events["JetNeutrinoMatched"] = self.events.JetNeutrinoMatched[
                    ~ak.is_none(self.events.JetNeutrinoMatched, axis=1)
                ]
                self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                    self.events.GenJetNeutrinoMatched,
                    self.events.JetNeutrinoMatched.eta,
                    "RecoEta",
                )
                self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                    self.events.MatchedJetsNeutrino,
                    self.events.JetNeutrinoMatched.phi,
                    "RecoPhi",
                )

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
                if int(os.environ.get("SPLITPNETREG15", 0)) == 1:
                    self.events[f"MatchedJetsSplit15"] = ak.with_field(
                        self.events.MatchedJets,
                        ak.where(
                            self.events.MatchedJets.JetPtRaw > 15,
                            0,
                            self.events.MatchedJets.ResponsePNetReg,
                        ),
                        "ResponsePNetReg",
                    )
                    self.events[f"MatchedJetsSplit15"] = ak.with_field(
                        self.events.MatchedJetsSplit15,
                        ak.where(
                            self.events.MatchedJets.JetPtRaw > 15,
                            0,
                            self.events.MatchedJets.JetPtPNetReg,
                        ),
                        "JetPtPNetReg",
                    )
                    self.events[f"MatchedJets"] = ak.with_field(
                        self.events.MatchedJets,
                        ak.where(
                            self.events.MatchedJets.JetPtRaw < 15,
                            0,
                            self.events.MatchedJets.ResponsePNetReg,
                        ),
                        "ResponsePNetReg",
                    )
                    self.events[f"MatchedJets"] = ak.with_field(
                        self.events.MatchedJets,
                        ak.where(
                            self.events.MatchedJets.JetPtRaw < 15,
                            0,
                            self.events.MatchedJets.JetPtPNetReg,
                        ),
                        "JetPtPNetReg",
                    )
                elif int(os.environ.get("PNETREG15", 0)) == 0:
                    self.events[f"MatchedJets"] = ak.with_field(
                        self.events.MatchedJets,
                        ak.where(
                            self.events.MatchedJets.JetPtRaw < 15,
                            (
                                ak.where(self.events.MatchedJets.ResponseRaw < 1, 0, 2)
                                if self.SetRegResponseToZero
                                else self.events.MatchedJets.ResponseRaw
                            ),
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

                # PNetRegNeutrino
                if int(os.environ.get("NEUTRINO", 1)) == 1:
                    self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                        self.events.MatchedJetsNeutrino,
                        self.events.JetNeutrinoMatched.pt
                        * (1 - self.events.JetNeutrinoMatched.rawFactor),
                        "JetPtRaw",
                    )
                    self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                        self.events.MatchedJetsNeutrino,
                        self.events.MatchedJetsNeutrino.JetPtRaw
                        / self.events.GenJetNeutrinoMatched.pt,
                        "ResponseRaw",
                    )
                    self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                        self.events.MatchedJetsNeutrino,
                        self.events.MatchedJetsNeutrino.JetPtRaw
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

                    if int(os.environ.get("SPLITPNETREG15", 0)) == 1:
                        self.events[f"MatchedJetsNeutrinoSplit15"] = ak.with_field(
                            self.events.MatchedJetsNeutrino,
                            ak.where(
                                self.events.MatchedJetsNeutrino.JetPtRaw > 15,
                                0,
                                self.events.MatchedJetsNeutrino.ResponsePNetRegNeutrino,
                            ),
                            "ResponsePNetRegNeutrino",
                        )
                        self.events[f"MatchedJetsNeutrinoSplit15"] = ak.with_field(
                            self.events.MatchedJetsNeutrinoSplit15,
                            ak.where(
                                self.events.MatchedJetsNeutrino.JetPtRaw > 15,
                                0,
                                self.events.MatchedJetsNeutrino.JetPtPNetRegNeutrino,
                            ),
                            "JetPtPNetRegNeutrino",
                        )
                        self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                            self.events.MatchedJetsNeutrino,
                            ak.where(
                                self.events.MatchedJetsNeutrino.JetPtRaw < 15,
                                0,
                                self.events.MatchedJetsNeutrino.ResponsePNetRegNeutrino,
                            ),
                            "ResponsePNetRegNeutrino",
                        )
                        self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                            self.events.MatchedJetsNeutrino,
                            ak.where(
                                self.events.MatchedJetsNeutrino.JetPtRaw < 15,
                                0,
                                self.events.MatchedJetsNeutrino.JetPtPNetRegNeutrino,
                            ),
                            "JetPtPNetRegNeutrino",
                        )
                    # when regression is not valid
                    elif int(os.environ.get("PNETREG15", 0)) == 0:
                        self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                            self.events.MatchedJetsNeutrino,
                            ak.where(
                                self.events.MatchedJetsNeutrino.JetPtRaw < 15,
                                (
                                    ak.where(
                                        self.events.MatchedJetsNeutrino.JetPtRaw
                                        / self.events.GenJetNeutrinoMatched.pt
                                        < 1,
                                        0,
                                        2,
                                    )
                                    if self.SetRegResponseToZero
                                    else self.events.MatchedJetsNeutrino.ResponseRaw
                                ),
                                self.events.MatchedJetsNeutrino.ResponsePNetRegNeutrino,
                            ),
                            "ResponsePNetRegNeutrino",
                        )
                        self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                            self.events.MatchedJetsNeutrino,
                            ak.where(
                                self.events.MatchedJetsNeutrino.JetPtRaw < 15,
                                self.events.MatchedJetsNeutrino.JetPtRaw,
                                self.events.MatchedJetsNeutrino.JetPtPNetRegNeutrino,
                            ),
                            "JetPtPNetRegNeutrino",
                        )

                    self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                        self.events.MatchedJetsNeutrino,
                        ak.where(
                            abs(self.events.MatchedJetsNeutrino.RecoEta) > 4.7,
                            self.events.MatchedJetsNeutrino.JetPtRaw
                            / self.events.GenJetNeutrinoMatched.pt,
                            self.events.MatchedJetsNeutrino.ResponsePNetRegNeutrino,
                        ),
                        "ResponsePNetRegNeutrino",
                    )
                    self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                        self.events.MatchedJetsNeutrino,
                        ak.where(
                            abs(self.events.MatchedJetsNeutrino.RecoEta) > 4.7,
                            self.events.MatchedJetsNeutrino.JetPtRaw,
                            self.events.MatchedJetsNeutrino.JetPtPNetRegNeutrino,
                        ),
                        "JetPtPNetRegNeutrino",
                    )

                if self.mc_truth_corr_pnetreg:
                    mc_truth_corr_factor_pnetreg = self.get_mc_truth_corr(
                        self.mc_truth_corr_pnetreg,
                        self.events.MatchedJets.RecoEta,
                        self.events.MatchedJets.RecoPhi,
                        self.events.MatchedJets.JetPtPNetReg,
                    )
                    self.events[f"MatchedJets"] = ak.with_field(
                        self.events.MatchedJets,
                        self.events.MatchedJets.JetPtPNetReg
                        * mc_truth_corr_factor_pnetreg,
                        "JetPtPNetReg",
                    )
                    self.events[f"MatchedJets"] = ak.with_field(
                        self.events.MatchedJets,
                        self.events.MatchedJets.ResponsePNetReg
                        * mc_truth_corr_factor_pnetreg,
                        "ResponsePNetReg",
                    )
                    # self.events[f"MatchedJets"] = ak.with_field(
                    #     self.events.MatchedJets,
                    #     mc_truth_corr_factor_pnetreg,
                    #     "MCTruthCorrPNetReg",
                    # )

                if (
                    self.mc_truth_corr_pnetreg_neutrino
                    and int(os.environ.get("NEUTRINO", 1)) == 1
                ):
                    mc_truth_corr_factor_pnetreg_neutrino = self.get_mc_truth_corr(
                        self.mc_truth_corr_pnetreg_neutrino,
                        self.events.MatchedJetsNeutrino.RecoEta,
                        self.events.MatchedJetsNeutrino.RecoPhi,
                        self.events.MatchedJetsNeutrino.JetPtPNetRegNeutrino,
                    )
                    self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                        self.events.MatchedJetsNeutrino,
                        self.events.MatchedJetsNeutrino.JetPtPNetRegNeutrino
                        * mc_truth_corr_factor_pnetreg_neutrino,
                        "JetPtPNetRegNeutrino",
                    )
                    self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                        self.events.MatchedJetsNeutrino,
                        self.events.MatchedJetsNeutrino.ResponsePNetRegNeutrino
                        * mc_truth_corr_factor_pnetreg_neutrino,
                        "ResponsePNetRegNeutrino",
                    )
                    # self.events[f"MatchedJetsNeutrino"] = ak.with_field(
                    #     self.events.MatchedJetsNeutrino,
                    #     mc_truth_corr_factor_pnetreg_neutrino,
                    #     "MCTruthCorrPNetRegNeutrino",
                    # )

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
        pass
