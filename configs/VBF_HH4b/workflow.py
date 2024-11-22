import awkward as ak
from dask.distributed import get_worker

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.lib.deltaR_matching import object_matching

from custom_cut_functions import *
from custom_cuts import *
from prediction_selection import *
from vbf_matching import analyze_parton_from_vbf_quarks
#TODO import parton_matching_functions

class VBFHH4bbQuarkMatchingProcessor(BaseProcessorABC):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.dr_min = self.workflow_options["parton_jet_min_dR"]
        self.max_num_jets = self.workflow_options["max_num_jets"]
        self.which_bquark = self.workflow_options["which_bquark"]

    def apply_object_preselection(self, variation):
        self.events["Jet"] = ak.with_field(
            self.events.Jet,
            self.events.Jet.pt
            * (1 - self.events.Jet.rawFactor)
            * self.events.Jet.PNetRegPtRawCorr
            * self.events.Jet.PNetRegPtRawCorrNeutrino,
            "pt",
        )
        self.events.Jet = ak.with_field(
             self.events.Jet, ak.local_index(self.events.Jet, axis=1), "index"
        )

        self.events["JetGood"] = jet_selection_nopu(self.events, "Jet", self.params)

        self.events["JetVBF_matching"] = self.events.Jet
        self.events["JetVBF_matching"] = jet_selection_nopu(self.events, "JetVBF_matching", self.params)

        self.events["JetGoodVBF"] = self.events.Jet
        self.events["JetGoodVBF"] = jet_selection_nopu(self.events, "JetGoodVBF", self.params)
        #self.events["JetGoodVBF"] = self.events.JetGood[:, :2] #Keep only the first 2 jets for QvG selection

        self.events["JetVBF_generalSelection"] = self.events.Jet
        self.events["JetVBF_generalSelection"] = jet_selection_nopu(self.events, "JetVBF_generalSelection", self.params)

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

    def do_vbf_parton_matching(self, which_bquark):  # -> ak.Array:
        self.events.GenPart = ak.with_field(self.events.GenPart, ak.local_index(self.events.GenPart, axis=1), "index")
        genpart = self.events.GenPart

        isQuark = abs(genpart.pdgId) < 7
        isHard = genpart.hasFlags(["fromHardProcess"])

        quarks = genpart[isQuark & isHard]
        quarks = quarks[quarks.genPartIdxMother!=-1]

        quarks_mother = genpart[quarks.genPartIdxMother]
        quarks_mother_children = quarks_mother.children
        quarks_mother_children_isH = ak.sum((quarks_mother_children.pdgId == 25), axis=-1)==2
        vbf_quarks = quarks[quarks_mother_children_isH]

        children_idxG = ak.without_parameters(genpart.childrenIdxG, behavior={})
        children_idxG_flat = ak.flatten(children_idxG, axis=1)
        genpart_pdgId_flat = ak.flatten(ak.without_parameters(genpart.pdgId, behavior={}), axis=1)
        genpart_LastCopy_flat = ak.flatten(ak.without_parameters(genpart.hasFlags(["isLastCopy"]), behavior={}), axis=1)
        genpart_pt_flat = ak.flatten(ak.without_parameters(genpart.pt, behavior={}), axis=1)
        genparts_flat = ak.flatten(genpart)
        genpart_offsets = np.concatenate([[0],np.cumsum(ak.to_numpy(ak.num(genpart, axis=1), allow_missing=True))])
        vbf_quark_idx = ak.to_numpy(vbf_quarks.index+genpart_offsets[:-1], allow_missing=False)
        vbf_quarks_pdgId = ak.to_numpy(vbf_quarks.pdgId, allow_missing=False)
        nevents=vbf_quark_idx.shape[0]
        firstgenpart_idxG = ak.firsts(genpart[:,0].children).genPartIdxMotherG
        firstgenpart_idxG_numpy = ak.to_numpy( firstgenpart_idxG, allow_missing=False)

        vbf_quark_last_idx=analyze_parton_from_vbf_quarks(
            vbf_quark_idx,
            vbf_quarks_pdgId,
            children_idxG_flat,
            genpart_pdgId_flat,
            genpart_offsets,
            genpart_LastCopy_flat,
            genpart_pt_flat,
            nevents,
            firstgenpart_idxG_numpy
        )

        vbf_quark_last = genparts_flat[vbf_quark_last_idx]

        matched_vbf_quarks, matched_vbf_jets, deltaR_matched_vbf = (
            object_matching(
                vbf_quark_last,
                self.events.JetVBF_matching,
                dr_min=self.dr_min,
            )
        )

        self.events["JetGoodVBF_matched"] = matched_vbf_jets
        self.events["quarkVBF_matched"] = matched_vbf_quarks

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
        self.events["nJetGoodVBF"] = ak.num(self.events.JetGoodVBF, axis=1)
        self.events["nJetVBF_generalSelection"] = ak.num(self.events.JetVBF_generalSelection, axis=1)

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

    def process_extra_after_presel(self, variation):# -> ak.Array:
        if self._isMC:
            self.do_parton_matching(which_bquark=self.which_bquark)
            # NOTE:  ak.num counts even the None values, while ak.count counts only the non-None values

            self.events["nbQuarkHiggsMatched"] = ak.num(
                self.events.bQuarkHiggsMatched, axis=1
            )
            self.events["nbQuarkMatched"] = ak.num(self.events.bQuarkMatched, axis=1)

            # reconstruct the higgs candidates
            self.events["RecoHiggs1"], self.events["RecoHiggs2"] = (
                self.reconstruct_higgs_candidates(self.events.JetGoodMatched)
            )

            self.do_vbf_parton_matching(which_bquark=self.which_bquark)

            
            #Create new variable delta eta and invariant mass of the jets
            # JetGoodVBF_padded = ak.pad_none(self.events.JetGoodVBF, 2) #Adds none jets to events that have less than 2 jets
            # self.events["deltaEta"] = abs(JetGoodVBF_padded.eta[:,0] - JetGoodVBF_padded.eta[:,1])
            # self.events["jj_mass"]=(self.events.JetGoodVBF[:,0]+self.events.JetGoodVBF[:,1]).mass

            #Create new variable delta eta and invariant mass of the jets
            JetGoodVBF_matched_padded = ak.pad_none(self.events.JetGoodVBF_matched, 2) #Adds none jets to events that have less than 2 jets
            self.events["deltaEta_matched"] = abs(JetGoodVBF_matched_padded.eta[:,0] - JetGoodVBF_matched_padded.eta[:,1])

            self.events["jj_mass_matched"]=(JetGoodVBF_matched_padded[:,0]+JetGoodVBF_matched_padded[:,1]).mass

            self.events["etaProduct"] = ((self.events.JetGoodVBF_matched.eta[:,0] * self.events.JetGoodVBF_matched.eta[:,1]) 
                                                     / abs(self.events.JetGoodVBF_matched.eta[:,0] * self.events.JetGoodVBF_matched.eta[:,1]))
    
        else:
            self.dummy_provenance()

        self.events["nJetGoodHiggsMatched"] = ak.num(
            self.events.JetGoodHiggsMatched, axis=1
        )
        self.events["nJetGoodMatched"] = ak.num(self.events.JetGoodMatched, axis=1)
