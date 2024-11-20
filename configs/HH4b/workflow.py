import awkward as ak
from dask.distributed import get_worker

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.lib.deltaR_matching import object_matching

from custom_cut_functions import *
from custom_cuts import *
from prediction_selection import *


class HH4bbQuarkMatchingProcessor(BaseProcessorABC):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.dr_min = self.workflow_options["parton_jet_min_dR"]
        self.max_num_jets = self.workflow_options["max_num_jets"]
        self.which_bquark = self.workflow_options["which_bquark"]
        self.fifth_jet = self.workflow_options["fifth_jet"]
        self.tight_cuts = self.workflow_options["tight_cuts"]
        self.classification = self.workflow_options["classification"]
        self.spanet_model = self.workflow_options["spanet_model"]

    def apply_object_preselection(self, variation):
        self.events["Jet"] = ak.with_field(
            self.events.Jet,
            self.events.Jet.pt
            * (1 - self.events.Jet.rawFactor)
            * self.events.Jet.PNetRegPtRawCorr
            * self.events.Jet.PNetRegPtRawCorrNeutrino,
            "pt",
        )
        self.events["Jet"] = ak.with_field(
            self.events.Jet,
            self.events.Jet.mass
            * (1 - self.events.Jet.rawFactor)
            * self.events.Jet.PNetRegPtRawCorr
            * self.events.Jet.PNetRegPtRawCorrNeutrino,
            "mass",
        )
        self.events["JetGood"] = jet_selection_nopu(
            self.events, "Jet", self.params, tight_cuts=self.tight_cuts
        )

        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
        )
        self.events["MuonGood"] = lepton_selection(self.events, "Muon", self.params)
        # order jet by btag score and keep only the first 4
        self.events["JetGood"] = self.events.JetGood[
            ak.argsort(self.events.JetGood.btagPNetB, axis=1, ascending=False)
        ]
        # keep only the first 4 jets for the Higgs candidates reconstruction
        self.events["JetGoodHiggs"] = self.events.JetGood[:, :4]

        # Trying to reshuffle jets 4 and above by pt instead of b-tag score
        if self.fifth_jet == "pt":
            self.events["JetGoodNoHiggs"] = self.events.JetGood[:, 4:]
            self.events["JetGoodNoHiggsPt"] = self.events.JetGoodNoHiggs[
                ak.argsort(self.events.JetGoodNoHiggs.pt, axis=1, ascending=False)
            ]
            self.events["JetGood"] = ak.concatenate(
                (self.events["JetGoodHiggs"], self.events["JetGoodNoHiggsPt"]), axis=1
            )
    #        five_pt  = ak.fill_none(ak.firsts(self.events["JetGoodNoHiggsPt"].pt ),value=9999)
    #        five_eta = ak.fill_none(ak.firsts(self.events["JetGoodNoHiggsPt"].eta),value=9999)
    #        five_phi = ak.fill_none(ak.firsts(self.events["JetGoodNoHiggsPt"].phi),value=9999)
    #        self.events["FifthJet"] = ak.with_field(
    #                                ak.firsts(self.events.JetGoodNoHiggsPt,
    #                                five_pt,
    #                                "pt",
    #                                ))
    #        self.events["FifthJet"] = ak.with_field(
    #                                ak.firsts(self.events.JetGoodNoHiggsPt,
    #                                five_eta,
    #                                "eta",
    #                                ))
    #        self.events["FifthJet"] = ak.with_field(
    #                                ak.firsts(self.events.JetGoodNoHiggsPt,
    #                                five_phi,
    #                                "phi",
    #                                ))
    #        breakpoint()

    def get_pairing_information(self, session, input_name, output_name):

        pt = np.array(
            np.log(
                ak.to_numpy(
                    ak.fill_none(
                        ak.pad_none(
                            self.events.JetGood.pt, self.max_num_jets, clip=True
                        ),
                        value=0,
                    ),
                    allow_missing=True,
                )
                + 1
            ),
            dtype=np.float32,
        )

        eta = np.array(
            ak.to_numpy(
                ak.fill_none(
                    ak.pad_none(self.events.JetGood.eta, self.max_num_jets, clip=True),
                    value=0,
                ),
                allow_missing=True,
            ),
            dtype=np.float32,
        )

        phi = np.array(
            ak.to_numpy(
                ak.fill_none(
                    ak.pad_none(self.events.JetGood.phi, self.max_num_jets, clip=True),
                    value=0,
                ),
                allow_missing=True,
            ),
            dtype=np.float32,
        )

        btag = np.array(
            ak.to_numpy(
                ak.fill_none(
                    ak.pad_none(
                        self.events.JetGood.btagPNetB, self.max_num_jets, clip=True
                    ),
                    value=0,
                ),
                allow_missing=True,
            ),
            dtype=np.float32,
        )

        mask = np.array(
            ak.to_numpy(
                ak.fill_none(
                    ak.pad_none(
                        ak.ones_like(self.events.JetGood.pt),
                        self.max_num_jets,
                        clip=True,
                    ),
                    value=0,
                ),
                allow_missing=True,
            ),
            dtype=np.bool_,
        )

        inputs = np.stack((pt, eta, phi, btag), axis=-1)
        inputs_complete = {input_name[0]: inputs, input_name[1]: mask}

        outputs = session.run(output_name, inputs_complete)

        # extract the best jet assignment from
        # the predicted probabilities
        assignment_probability = np.stack((outputs[0], outputs[1]), axis=0)
        # print("\nassignment_probability", assignment_probability)
        # swap axis
        predictions_best = np.swapaxes(
            extract_predictions(assignment_probability), 0, 1
        )

        # get the probabilities of the best jet assignment
        num_events = assignment_probability.shape[1]
        range_num_events = np.arange(num_events)
        best_pairing_probabilities = np.ndarray((2, num_events))
        for i in range(2):
            best_pairing_probabilities[i] = assignment_probability[
                i,
                range_num_events,
                predictions_best[:, i, 0],
                predictions_best[:, i, 1],
            ]
        best_pairing_probabilities_sum = np.sum(best_pairing_probabilities, axis=0)
        # print("\nbest_pairing_probabilities_sum", best_pairing_probabilities_sum)

        # set to zero the probabilities of the best jet assignment, the symmetrization and the same jet assignment on the other target
        for j in range(2):
            for k in range(2):
                assignment_probability[
                    j,
                    range_num_events,
                    predictions_best[:, j, k],
                    predictions_best[:, j, 1 - k],
                ] = 0
                assignment_probability[
                    1 - j,
                    range_num_events,
                    predictions_best[:, j, k],
                    predictions_best[:, j, 1 - k],
                ] = 0

        # print("\nassignment_probability new", assignment_probability)
        # extract the second best jet assignment from
        # the predicted probabilities
        # swap axis
        predictions_second_best = np.swapaxes(
            extract_predictions(assignment_probability), 0, 1
        )

        # get the probabilities of the second best jet assignment
        second_best_pairing_probabilities = np.ndarray((2, num_events))
        for i in range(2):
            second_best_pairing_probabilities[i] = assignment_probability[
                i,
                range_num_events,
                predictions_second_best[:, i, 0],
                predictions_second_best[:, i, 1],
            ]
        second_best_pairing_probabilities_sum = np.sum(
            second_best_pairing_probabilities, axis=0
        )
        # print(
        #     "\nsecond_best_pairing_probabilities_sum",
        #     second_best_pairing_probabilities_sum,
        # )

        return (
            predictions_best,
            best_pairing_probabilities_sum,
            second_best_pairing_probabilities_sum,
        )

    def reconstruct_higgs(self, jet_collection, idx_collection):
        higgs_1 = ak.unflatten(
            jet_collection[np.arange(len(idx_collection)), idx_collection[:, 0, 0]]
            + jet_collection[np.arange(len(idx_collection)), idx_collection[:, 0, 1]],
            1,
        )
        higgs_2 = ak.unflatten(
            jet_collection[np.arange(len(idx_collection)), idx_collection[:, 1, 0]]
            + jet_collection[np.arange(len(idx_collection)), idx_collection[:, 1, 1]],
            1,
        )

        higgs_leading_index = ak.where(higgs_1.pt > higgs_2.pt, 0, 1)

        higgs_lead = ak.where(higgs_leading_index == 0, higgs_1, higgs_2)
        higgs_sub = ak.where(higgs_leading_index == 0, higgs_2, higgs_1)

        higgs_leading_index_expanded = higgs_leading_index[
            :, np.newaxis, np.newaxis
        ] * np.ones((2, 2))
        idx_ordered = ak.where(
            higgs_leading_index_expanded == 0, idx_collection, idx_collection[:, ::-1]
        )[0]

        return higgs_lead, higgs_sub, idx_ordered

    def do_parton_matching(self, which_bquark):  # -> ak.Array:
        # Select b-quarks at Gen level, coming from H->bb decay
        self.events.GenPart = ak.with_field(
            self.events.GenPart, ak.local_index(self.events.GenPart, axis=1), "index"
        )
        isHiggs = self.events.GenPart.pdgId == 25
        isLast = self.events.GenPart.hasFlags(["isLastCopy"])
        isHard = self.events.GenPart.hasFlags(["fromHardProcess"])
        higgs = self.events.GenPart[isHiggs & isLast & isHard]

        higgs = higgs[ak.num(higgs.childrenIdxG, axis=2) == 2]

        higgs = higgs[ak.argsort(higgs.pt, ascending=False)]
        if which_bquark == "last":
            isB = abs(self.events.GenPart.pdgId) == 5
            bquarks = self.events.GenPart[isB & isLast & isHard]
            bquarks_first = bquarks
            while True:
                b_mother = self.events.GenPart[bquarks_first.genPartIdxMother]
                mask_mother = (abs(b_mother.pdgId) == 5) | ((b_mother.pdgId) == 25)
                bquarks = bquarks[mask_mother]
                bquarks_first = bquarks_first[mask_mother]
                b_mother = b_mother[mask_mother]
                if ak.all((b_mother.pdgId) == 25):
                    break
                bquarks_first = ak.where(
                    abs(b_mother.pdgId) == 5, b_mother, bquarks_first
                )
            provenance = ak.where(
                bquarks_first.genPartIdxMother == higgs.index[:, 0], 1, 2
            )
        elif which_bquark == "first":
            bquarks = ak.flatten(higgs.children, axis=2)
            provenance = ak.where(bquarks.genPartIdxMother == higgs.index[:, 0], 1, 2)
        else:
            raise ValueError(
                "which_bquark for the parton matching must be 'first' or 'last'"
            )

        # Adding the provenance to the quark object
        bquarks = ak.with_field(bquarks, provenance, "provenance")
        self.events["bQuark"] = bquarks

        # Calling our general object_matching function.
        # The output is an awkward array with the shape of the second argument and None where there is no matching.
        # So, calling like this, we will get out an array of matched_quarks with the dimension of the JetGood.
        matched_bquarks_higgs, matched_jets_higgs, deltaR_matched_higgs = (
            object_matching(
                bquarks,
                self.events.JetGoodHiggs,
                dr_min=self.dr_min,
            )
        )
        # matched all jetgood
        matched_bquarks, matched_jets, deltaR_matched = object_matching(
            bquarks,
            self.events.JetGood,
            dr_min=self.dr_min,
        )

        matched_jets_higgs = ak.with_field(
            matched_jets_higgs, matched_bquarks_higgs.provenance, "provenance"
        )
        self.events["JetGoodHiggs"] = ak.with_field(
            self.events.JetGoodHiggs, matched_bquarks_higgs.provenance, "provenance"
        )
        matched_jets = ak.with_field(
            matched_jets, matched_bquarks.provenance, "provenance"
        )
        self.events["JetGood"] = ak.with_field(
            self.events.JetGood, matched_bquarks.provenance, "provenance"
        )

        self.events["bQuarkHiggsMatched"] = ak.with_field(
            matched_bquarks_higgs, deltaR_matched_higgs, "dRMatchedJet"
        )
        self.events["JetGoodHiggsMatched"] = ak.with_field(
            matched_jets_higgs, deltaR_matched_higgs, "dRMatchedJet"
        )
        self.events["bQuarkMatched"] = ak.with_field(
            matched_bquarks, deltaR_matched, "dRMatchedJet"
        )
        self.events["JetGoodMatched"] = ak.with_field(
            matched_jets, deltaR_matched, "dRMatchedJet"
        )
        self.events["JetGoodHiggsMatched"] = ak.with_field(
            self.events.JetGoodHiggsMatched,
            self.events.bQuarkHiggsMatched.pdgId,
            "pdgId",
        )
        self.events["JetGoodMatched"] = ak.with_field(
            self.events.JetGoodMatched,
            self.events.bQuarkMatched.pdgId,
            "pdgId",
        )

    def dummy_provenance(self):
        self.events["JetGoodHiggs"] = ak.with_field(
            self.events.JetGoodHiggs,
            ak.ones_like(self.events.JetGoodHiggs.pt) * -1,
            "provenance",
        )
        self.events["JetGoodHiggsMatched"] = self.events.JetGoodHiggs

        self.events["JetGood"] = ak.with_field(
            self.events.JetGood, ak.ones_like(self.events.JetGood.pt) * -1, "provenance"
        )
        self.events["JetGoodMatched"] = self.events.JetGood

    def count_objects(self, variation):
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood, axis=1)
        self.events["nMuonGood"] = ak.num(self.events.MuonGood, axis=1)
        self.events["nJetGood"] = ak.num(self.events.JetGood, axis=1)
        self.events["nJetGoodHiggs"] = ak.num(self.events.JetGoodHiggs, axis=1)

    def reconstruct_higgs_candidates(self, matched_jets_higgs):

        jet_higgs1 = matched_jets_higgs[matched_jets_higgs.provenance == 1]
        jet_higgs2 = matched_jets_higgs[matched_jets_higgs.provenance == 2]

        reco_higgs1 = jet_higgs1[:, 0] + jet_higgs1[:, 1]
        reco_higgs2 = jet_higgs2[:, 0] + jet_higgs2[:, 1]
        reco_higgs1 = ak.with_field(reco_higgs1, reco_higgs1.pt, "pt")
        reco_higgs2 = ak.with_field(reco_higgs2, reco_higgs2.pt, "pt")
        reco_higgs1 = ak.with_field(reco_higgs1, reco_higgs1.eta, "eta")
        reco_higgs2 = ak.with_field(reco_higgs2, reco_higgs2.eta, "eta")
        reco_higgs1 = ak.with_field(reco_higgs1, reco_higgs1.phi, "phi")
        reco_higgs2 = ak.with_field(reco_higgs2, reco_higgs2.phi, "phi")
        reco_higgs1 = ak.with_field(reco_higgs1, reco_higgs1.mass, "mass")
        reco_higgs2 = ak.with_field(reco_higgs2, reco_higgs2.mass, "mass")

        return reco_higgs1, reco_higgs2

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        if self._isMC and not self.classification:
            self.do_parton_matching(which_bquark=self.which_bquark)
            # NOTE:  ak.num counts even the None values, while ak.count counts only the non-None values

            self.events["nbQuarkHiggsMatched"] = ak.num(
                self.events.bQuarkHiggsMatched, axis=1
            )
            self.events["nbQuarkMatched"] = ak.num(self.events.bQuarkMatched, axis=1)

            # reconstruct the higgs candidates
            self.events["RecoHiggs1"], self.events["RecoHiggs2"] = (
                self.reconstruct_higgs_candidates(self.events.JetGoodHiggsMatched)
            )

        elif self.classification:
            self.dummy_provenance()

            try:
                worker = get_worker()
                # print("found worker", worker)
            except ValueError:
                worker = None
                # print("     >>>>>>>>>>   NOT found worker", worker)

            if worker is None:
                import onnxruntime as ort

                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                model_session = ort.InferenceSession(
                    self.spanet_model,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
                # input_name = [input.name for input in model_session.get_inputs()]
                # output_name = [output.name for output in model_session.get_outputs()]
            # print("     >>>>>>>>>>   initialize new worker", worker)
            else:
                model_session = worker.data["model_session"]
                # input_name = worker.data['input_name']
                # output_name = worker.data['output_name']
                # print("get info from old worker", worker)

            input_name = [input.name for input in model_session.get_inputs()]
            output_name = [output.name for output in model_session.get_outputs()]
            # print(model_session)
            (
                pairing_predictions,
                self.events["best_pairing_probability"],
                self.events["second_best_pairing_probability"],
            ) = self.get_pairing_information(model_session, input_name, output_name)

            # get the probabilities difference between the best and second best jet assignment
            self.events["Delta_pairing_probabilities"] = (
                self.events.best_pairing_probability
                - self.events.second_best_pairing_probability
            )
            # print("\nDelta_pairing_probabilities", self.events["Delta_pairing_probabilities"])

            # ADDITIONAL VARIABLES

            # HT : scalar sum of all jets with pT > 25 GeV inside | η | < 2.5
            self.events["HT"] = ak.sum(self.events.JetGood.pt, axis=1)

            # Minimum ∆R ( jj ) among all possible pairings of the leading b-tagged jets
            # Maximum ∆R( jj ) among all possible pairings of the leading b-tagged jets
            _, JetGood2 = ak.unzip(
                ak.cartesian(
                    [
                        self.events.JetGood[:, : self.max_num_jets],
                        self.events.JetGood[:, : self.max_num_jets],
                    ],
                    nested=True,
                )
            )
            dR = self.events.JetGood[:, : self.max_num_jets].delta_r(JetGood2)
            # remove dR between the same jets
            dR = ak.mask(dR, dR > 0)
            # flatten the last 2 dimension of the dR array  to get an array for each event
            dR = ak.flatten(dR, axis=2)
            self.events["dR_min"] = ak.min(dR, axis=1)
            self.events["dR_max"] = ak.max(dR, axis=1)

            # Leading-pT H candidate pT , η, φ, and mass
            # Subleading-pT H candidate pT , η, φ, and mass
            (
                self.events["HiggsLeading"],
                self.events["HiggsSubLeading"],
                pairing_predictions_ordered,
            ) = self.reconstruct_higgs(self.events.JetGood, pairing_predictions)

            self.events["HiggsLeading"] = ak.with_field(
                self.events.HiggsLeading, self.events.HiggsLeading.pt, "pt"
            )
            self.events["HiggsSubLeading"] = ak.with_field(
                self.events.HiggsSubLeading, self.events.HiggsSubLeading.pt, "pt"
            )
            self.events["HiggsLeading"] = ak.with_field(
                self.events.HiggsLeading, self.events.HiggsLeading.eta, "eta"
            )
            self.events["HiggsSubLeading"] = ak.with_field(
                self.events.HiggsSubLeading, self.events.HiggsSubLeading.eta, "eta"
            )
            self.events["HiggsLeading"] = ak.with_field(
                self.events.HiggsLeading, self.events.HiggsLeading.phi, "phi"
            )
            self.events["HiggsSubLeading"] = ak.with_field(
                self.events.HiggsSubLeading, self.events.HiggsSubLeading.phi, "phi"
            )
            self.events["HiggsLeading"] = ak.with_field(
                self.events.HiggsLeading, self.events.HiggsLeading.mass, "mass"
            )
            self.events["HiggsSubLeading"] = ak.with_field(
                self.events.HiggsSubLeading, self.events.HiggsSubLeading.mass, "mass"
            )

            # Angular separation (∆R) between b jets for each H candidate
            self.events["HiggsLeading"] = ak.with_field(
                self.events.HiggsLeading,
                self.events.JetGood[
                    np.arange(len(pairing_predictions_ordered)),
                    pairing_predictions_ordered[:, 0, 0],
                ].delta_r(
                    self.events.JetGood[
                        np.arange(len(pairing_predictions_ordered)),
                        pairing_predictions_ordered[:, 0, 1],
                    ]
                ),
                "dR",
            )
            self.events["HiggsSubLeading"] = ak.with_field(
                self.events.HiggsSubLeading,
                self.events.JetGood[
                    np.arange(len(pairing_predictions_ordered)),
                    pairing_predictions_ordered[:, 1, 0],
                ].delta_r(
                    self.events.JetGood[
                        np.arange(len(pairing_predictions_ordered)),
                        pairing_predictions_ordered[:, 1, 1],
                    ]
                ),
                "dR",
            )

            # TODO change the definition
            # helicity | cos θ | for each H candidate
            self.events["HiggsLeading"] = ak.with_field(
                self.events.HiggsLeading,
                abs(np.cos(self.events.HiggsLeading.theta)),
                "cos_theta",
            )
            self.events["HiggsSubLeading"] = ak.with_field(
                self.events.HiggsSubLeading,
                abs(np.cos(self.events.HiggsSubLeading.theta)),
                "cos_theta",
            )

            # di-Higgs system
            # pT , η, and mass of HH system
            self.events["HH"] = self.events.HiggsLeading + self.events.HiggsSubLeading

            self.events["HH"] = ak.with_field(self.events.HH, self.events.HH.pt, "pt")
            self.events["HH"] = ak.with_field(self.events.HH, self.events.HH.eta, "eta")
            self.events["HH"] = ak.with_field(
                self.events.HH, self.events.HH.mass, "mass"
            )

            # TODO change the definition
            # | cos θ ∗ | of HH system
            self.events["HH"] = ak.with_field(
                self.events.HH, abs(np.cos(self.events.HH.theta)), "cos_theta_star"
            )

            # Angular separation (∆R, ∆η, ∆φ) between H candidates
            self.events["HH"] = ak.with_field(
                self.events.HH,
                self.events.HiggsLeading.delta_r(self.events.HiggsSubLeading),
                "dR",
            )
            self.events["HH"] = ak.with_field(
                self.events.HH,
                abs(self.events.HiggsLeading.eta - self.events.HiggsSubLeading.eta),
                "dEta",
            )
            self.events["HH"] = ak.with_field(
                self.events.HH,
                self.events.HiggsLeading.delta_phi(self.events.HiggsSubLeading),
                "dPhi",
            )

        else:
            self.dummy_provenance()

        self.events["nJetGoodHiggsMatched"] = ak.num(
            self.events.JetGoodHiggsMatched, axis=1
        )
        self.events["nJetGoodMatched"] = ak.num(self.events.JetGoodMatched, axis=1)


# TODO:
# NEW THETA ANGLE DEFINITION

# float HelicityCosTheta(TLorentzVector Booster, TLorentzVector Boosted)
# {
#     TVector3 BoostVector = Booster.BoostVector();
#     Boosted.Boost( -BoostVector.x(), -BoostVector.y(), -BoostVector.z() );
#     return Boosted.CosTheta();
# }

# higgs1_helicityCosTheta =fabs(HelicityCosTheta( leadingHiggsCands.at(best_pairing_index)  ,  leading_higgs_leading_jet ));
# higgs2_helicityCosTheta =fabs(HelicityCosTheta( subleadingHiggsCands.at(best_pairing_index), subleading_higgs_leading_jet ));

# //Costhetastar in CS frame
# TLorentzVector higgs1_vec;
# higgs1_vec =  leadingHiggsCands.at(best_pairing_index);
# higgs1_vec.Boost( - hh_vec.BoostVector());
# hh_CosThetaStar_CS = fabs(higgs1_vec.CosTheta());
