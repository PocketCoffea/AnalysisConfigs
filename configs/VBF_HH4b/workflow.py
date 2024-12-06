import awkward as ak
from dask.distributed import get_worker
import sys

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.lib.deltaR_matching import object_matching

from custom_cut_functions import *
from custom_cuts import *

sys.path.append("../../")
from utils.parton_matching_function import get_parton_last_copy
from utils.spanet_evaluation_functions import get_pairing_information, get_best_pairings
from utils.basic_functions import add_fields
from utils.reconstruct_higgs_candidates import (
    reconstruct_higgs_from_provenance,
    reconstruct_higgs_from_idx,
)
from vbf_matching import get_jets_no_higgs


class VBFHH4bbQuarkMatchingProcessor(BaseProcessorABC):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.dr_min = self.workflow_options["parton_jet_min_dR"]
        self.max_num_jets = self.workflow_options["max_num_jets"]
        self.which_bquark = self.workflow_options["which_bquark"]
        self.spanet_model = self.workflow_options["spanet_model"]
        self.vbf_parton_matching = self.workflow_options["vbf_parton_matching"]

    def apply_object_preselection(self, variation):
        self.events["Jet"] = ak.with_field(
            self.events.Jet,
            ak.where(
                self.events.Jet.PNetRegPtRawCorr > 0,
                self.events.Jet.pt
                * (1 - self.events.Jet.rawFactor)
                * self.events.Jet.PNetRegPtRawCorr,
                self.events.Jet.pt,
            ),
            "pt",
        )
        self.events["Jet"] = ak.with_field(
            self.events.Jet,
            ak.where(
                self.events.Jet.PNetRegPtRawCorr > 0,
                self.events.Jet.mass
                * (1 - self.events.Jet.rawFactor)
                * self.events.Jet.PNetRegPtRawCorr,
                self.events.Jet.mass,
            ),
            "mass",
        )
        self.events["Jet"] = ak.with_field(
            self.events.Jet, ak.local_index(self.events.Jet, axis=1), "index"
        )

        self.events["JetGood"] = self.events.Jet
        self.events["JetGood"] = ak.with_field(
            self.events.JetGood,
            ak.where(
                self.events.JetGood.PNetRegPtRawCorr > 0,
                self.events.JetGood.pt * self.events.JetGood.PNetRegPtRawCorrNeutrino,
                self.events.JetGood.pt,
            ),
            "pt",
        )
        self.events["JetGood"] = ak.with_field(
            self.events.JetGood,
            ak.where(
                self.events.JetGood.PNetRegPtRawCorr > 0,
                self.events.JetGood.mass * self.events.JetGood.PNetRegPtRawCorrNeutrino,
                self.events.JetGood.mass,
            ),
            "mass",
        )

        self.events["JetGood"] = jet_selection_nopu(self.events, "JetGood", self.params)

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

        self.events["JetGoodHiggsPtOrder"] = self.events.JetGoodHiggs[
            ak.argsort(self.events.JetGoodHiggs.pt, axis=1, ascending=False)
        ]

    def get_jet_higgs_provenance(self, which_bquark):  # -> ak.Array:
        # Select b-quarks at Gen level, coming from H->bb decay
        self.events["GenPart"] = ak.with_field(
            self.events.GenPart, ak.local_index(self.events.GenPart, axis=1), "index"
        )
        genpart = self.events.GenPart

        isHiggs = genpart.pdgId == 25
        isB = abs(genpart.pdgId) == 5
        isLast = genpart.hasFlags(["isLastCopy"])
        isFirst = genpart.hasFlags(["isFirstCopy"])
        isHard = genpart.hasFlags(["fromHardProcess"])

        higgs = genpart[isHiggs & isLast & isHard]
        higgs = higgs[ak.num(higgs.childrenIdxG, axis=2) == 2]
        higgs = higgs[ak.argsort(higgs.pt, ascending=False)]
        higgs = higgs[ak.argsort(higgs.pt, ascending=False)]

        if which_bquark == "last_numba":
            bquarks_first = genpart[isB & isHard & isFirst]
            mother_bquarks = genpart[bquarks_first.genPartIdxMother]
            bquarks_from_higgs = bquarks_first[mother_bquarks.pdgId == 25]
            provenance = ak.where(
                bquarks_from_higgs.genPartIdxMother == higgs.index[:, 0], 1, 2
            )

            # define variables to get the last copy
            children_idxG = ak.without_parameters(genpart.childrenIdxG, behavior={})
            children_idxG_flat = ak.flatten(children_idxG, axis=1)
            genpart_pdgId_flat = ak.flatten(
                ak.without_parameters(genpart.pdgId, behavior={}), axis=1
            )
            genpart_LastCopy_flat = ak.flatten(
                ak.without_parameters(genpart.hasFlags(["isLastCopy"]), behavior={}),
                axis=1,
            )
            genpart_pt_flat = ak.flatten(
                ak.without_parameters(genpart.pt, behavior={}), axis=1
            )
            genparts_flat = ak.flatten(genpart)
            genpart_offsets = np.concatenate(
                [
                    [0],
                    np.cumsum(ak.to_numpy(ak.num(genpart, axis=1), allow_missing=True)),
                ]
            )
            b_quark_idx = ak.to_numpy(
                bquarks_from_higgs.index + genpart_offsets[:-1], allow_missing=False
            )
            b_quarks_pdgId = ak.to_numpy(bquarks_from_higgs.pdgId, allow_missing=False)
            nevents = b_quark_idx.shape[0]
            firstgenpart_idxG = ak.firsts(genpart[:, 0].children).genPartIdxMotherG
            firstgenpart_idxG_numpy = ak.to_numpy(
                firstgenpart_idxG, allow_missing=False
            )

            b_quark_last_idx = get_parton_last_copy(
                b_quark_idx,
                b_quarks_pdgId,
                children_idxG_flat,
                genpart_pdgId_flat,
                genpart_offsets,
                genpart_LastCopy_flat,
                genpart_pt_flat,
                nevents,
                firstgenpart_idxG_numpy,
            )
            bquarks = genparts_flat[b_quark_last_idx]

        elif which_bquark == "last":
            bquarks = genpart[isB & isLast & isHard]
            bquarks_first = bquarks
            while True:
                b_mother = genpart[bquarks_first.genPartIdxMother]
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
                "which_bquark for the parton matching must be 'first' or 'last' or 'last_numba'"
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

    def do_vbf_parton_matching(self, which_bquark):  # -> ak.Array:
        # Select vbf quarks
        self.events.GenPart = ak.with_field(
            self.events.GenPart, ak.local_index(self.events.GenPart, axis=1), "index"
        )
        genpart = self.events.GenPart

        isQuark = abs(genpart.pdgId) < 7
        isHard = genpart.hasFlags(["fromHardProcess"])

        quarks = genpart[isQuark & isHard]
        quarks = quarks[quarks.genPartIdxMother != -1]

        quarks_mother = genpart[quarks.genPartIdxMother]
        quarks_mother_children = quarks_mother.children
        quarks_mother_children_isH = (
            ak.sum((quarks_mother_children.pdgId == 25), axis=-1) == 2
        )
        vbf_quarks = quarks[quarks_mother_children_isH]

        children_idxG = ak.without_parameters(genpart.childrenIdxG, behavior={})
        children_idxG_flat = ak.flatten(children_idxG, axis=1)
        genpart_pdgId_flat = ak.flatten(
            ak.without_parameters(genpart.pdgId, behavior={}), axis=1
        )
        genpart_LastCopy_flat = ak.flatten(
            ak.without_parameters(genpart.hasFlags(["isLastCopy"]), behavior={}), axis=1
        )
        genpart_pt_flat = ak.flatten(
            ak.without_parameters(genpart.pt, behavior={}), axis=1
        )
        genparts_flat = ak.flatten(genpart)
        genpart_offsets = np.concatenate(
            [[0], np.cumsum(ak.to_numpy(ak.num(genpart, axis=1), allow_missing=True))]
        )
        vbf_quark_idx = ak.to_numpy(
            vbf_quarks.index + genpart_offsets[:-1], allow_missing=False
        )
        vbf_quarks_pdgId = ak.to_numpy(vbf_quarks.pdgId, allow_missing=False)
        nevents = vbf_quark_idx.shape[0]
        firstgenpart_idxG = ak.firsts(genpart[:, 0].children).genPartIdxMotherG
        firstgenpart_idxG_numpy = ak.to_numpy(firstgenpart_idxG, allow_missing=False)

        vbf_quark_last_idx = get_parton_last_copy(
            vbf_quark_idx,
            vbf_quarks_pdgId,
            children_idxG_flat,
            genpart_pdgId_flat,
            genpart_offsets,
            genpart_LastCopy_flat,
            genpart_pt_flat,
            nevents,
            firstgenpart_idxG_numpy,
        )

        vbf_quark_last = genparts_flat[vbf_quark_last_idx]

        matched_vbf_quarks, matched_vbf_jets, deltaR_matched_vbf = object_matching(
            vbf_quark_last,
            self.events.JetVBF_matching,
            dr_min=self.dr_min,
        )

        maskNotNone = ~ak.is_none(matched_vbf_jets, axis=1)
        self.events["JetVBF_matched"] = matched_vbf_jets[maskNotNone]

        self.events["JetVBF_matched"] = ak.with_field(
            self.events.JetVBF_matched,
            ak.where(
                self.events.JetVBF_matched.PNetRegPtRawCorr > 0,
                self.events.JetVBF_matched.pt
                / self.events.JetVBF_matched.PNetRegPtRawCorrNeutrino,
                self.events.JetVBF_matched.pt,
            ),
            "pt",
        )

        self.events["quarkVBF_matched"] = matched_vbf_quarks[maskNotNone]

        self.events["quarkVBF"] = vbf_quark_last

        # general Selection
        (
            matched_vbf_quarks_generalSelection,
            matched_vbf_jets_generalSelection,
            deltaR_matched_vbf,
        ) = object_matching(
            vbf_quark_last,
            self.events.JetVBF_generalSelection,
            dr_min=self.dr_min,
        )
        maskNotNone_genSel = ~ak.is_none(matched_vbf_jets_generalSelection, axis=1)

        self.events["JetVBF_generalSelection_matched"] = (
            matched_vbf_jets_generalSelection[maskNotNone_genSel]
        )

        self.events["quarkVBF_generalSelection_matched"] = (
            matched_vbf_quarks_generalSelection[maskNotNone_genSel]
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
        # NOTE:  ak.num counts even the None values, while ak.count counts only the non-None values
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood, axis=1)
        self.events["nMuonGood"] = ak.num(self.events.MuonGood, axis=1)
        self.events["nJetGood"] = ak.num(self.events.JetGood, axis=1)
        self.events["nJetGoodHiggs"] = ak.num(self.events.JetGoodHiggs, axis=1)
        self.events["nJetGoodVBF"] = ak.num(self.events.JetGoodVBF, axis=1)
        self.events["nJetVBF_generalSelection"] = ak.num(
            self.events.JetVBF_generalSelection, axis=1
        )

    def process_extra_after_presel(self, variation):  # -> ak.Array:
        if self._isMC:
            if not self.spanet_model:
                # do truth matching to get b-jet from Higgs
                self.get_jet_higgs_provenance(which_bquark=self.which_bquark)
                self.events["nbQuarkHiggsMatched"] = ak.num(
                    self.events.bQuarkHiggsMatched, axis=1
                )
                self.events["nbQuarkMatched"] = ak.num(
                    self.events.bQuarkMatched, axis=1
                )

                # reconstruct the higgs candidates
                (
                    self.events["HiggsLeading"],
                    self.events["HiggsSubLeading"],
                    self.events["JetGoodFromHiggsOrdered"],
                ) = reconstruct_higgs_from_provenance(self.events.JetGoodMatched)

                matched_jet_higgs_idx_not_none = self.events.JetGoodMatched.index[
                    ~ak.is_none(self.events.JetGoodMatched.index, axis=1)
                ]
            else:
                # apply spanet model to get the pairing prediction for the b-jets from Higgs
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

                # compute the pairing information using the spanet model
                outputs = get_pairing_information(
                    model_session,
                    input_name,
                    output_name,
                    self.events,
                    self.max_num_jets,
                )

                (
                    pairing_predictions,
                    self.events["best_pairing_probability"],
                    self.events["second_best_pairing_probability"],
                ) = get_best_pairings(outputs)

                (
                    self.events["HiggsLeading"],
                    self.events["HiggsSubLeading"],
                    self.events["JetGoodFromHiggsOrdered"],
                ) = reconstruct_higgs_from_idx(self.events.JetGood, pairing_predictions)

                matched_jet_higgs_idx_not_none = self.events.JetGoodFromHiggsOrdered.index


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
                ak.flatten(matched_jet_higgs_idx_not_none + jet_offsets[:-1]),
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
            self.events["JetVBFNotFromHiggs"]=jet_selection_nopu(self.events, "JetVBFNotFromHiggs", self.params)

            # order in pt
            self.events["JetVBFNotFromHiggs"] = self.events.JetVBFNotFromHiggs[
                ak.argsort(self.events.JetVBFNotFromHiggs.pt, axis=1, ascending=False)
            ]

            self.events["HH"] = add_fields(
                self.events.HiggsLeading + self.events.HiggsSubLeading
            )

            if self.vbf_parton_matching:
                self.do_vbf_parton_matching(which_bquark=self.which_bquark)

            # num_JetVBF_matched = ak.num(self.events.JetVBF_matched)
            # num_eventsTwoVBF = ak.sum((num_JetVBF_matched == 2))

            # num_eventsOneVBF = ak.sum((num_JetVBF_matched == 1))
            # num_eventsZeroVBF = ak.sum((num_JetVBF_matched == 0))
            # print(len(self.events))
            # print("Two", num_eventsTwoVBF/len(self.events.JetVBF_matched))
            # print("One", num_eventsOneVBF/len(self.events.JetVBF_matched))
            # print("Zero", num_eventsZeroVBF/len(self.events.JetVBF_matched))
            # print(ak.sum(num_JetVBF_matched) / (len(self.events) * 2))

            self.events["nJetVBF_matched"] = ak.num(
                self.events.JetVBF_matched, axis=1
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
                ],1
            )
            vbf_jets_max_mass_1 = ak.unflatten(
                self.events.Jet[
                    ak.local_index(self.events.Jet, axis=0),
                    ak.to_numpy(jets_max_mass["1"].index),
                ],1
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

            # Create new variable delta eta and invariant mass of the jets
            JetVBF_matched_padded = ak.pad_none(
                self.events.JetVBF_matched, 2
            )  # Adds none jets to events that have less than 2 jets

            self.events["deltaEta_matched"] = abs(
                JetVBF_matched_padded.eta[:, 0]
                - JetVBF_matched_padded.eta[:, 1]
            )

            self.events["jj_mass_matched"] = (
                JetVBF_matched_padded[:, 0] + JetVBF_matched_padded[:, 1]
            ).mass

            # This product will give only -1 or 1 values, as it's needed to see if the two jets are in the same side or not
            self.events["etaProduct"] = (
                JetVBF_matched_padded.eta[:, 0]
                * JetVBF_matched_padded.eta[:, 1]
            ) / abs(
                JetVBF_matched_padded.eta[:, 0]
                * JetVBF_matched_padded.eta[:, 1]
            )

        else:
            self.dummy_provenance()

        self.events["nJetGoodHiggsMatched"] = ak.num(
            self.events.JetGoodHiggsMatched, axis=1
        )
        self.events["nJetGoodMatched"] = ak.num(self.events.JetGoodMatched, axis=1)
