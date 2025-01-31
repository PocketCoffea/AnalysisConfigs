import sys
import awkward as ak
import numba

from pocket_coffea.workflows.tthbb_base_processor import ttHbbBaseProcessor
from pocket_coffea.lib.deltaR_matching import object_matching
from pocket_coffea.lib.parton_provenance import *


class PartonMatchingProcessorWithFSR(ttHbbBaseProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.dr_min = self.workflow_options["parton_jet_min_dR"]
        self.dr_min_postfsr = self.workflow_options.get("parton_jet_min_dR_postfsr", 1.)

    def do_parton_matching_ttHbb(self) -> ak.Array:

        genparts = self.events.GenPart
        
        children_idxG = ak.without_parameters(genparts.childrenIdxG, behavior={})
        children_idxG_flat = ak.flatten(children_idxG, axis=1)
        genparts_flat = ak.flatten(genparts)
        genparts_offsets = np.concatenate([[0],np.cumsum(ak.to_numpy(ak.num(genparts, axis=1), allow_missing=True))])
        local_index_all = ak.local_index(genparts, axis=1)

        # Get the initial partons, first copy
        initial = genparts.genPartIdxMother == 0
        hard_process = (genparts.hasFlags(['fromHardProcess','isPrompt','isHardProcess', 'isFirstCopy'])) & (genparts.status !=21) # exclude incoming particles
        genparts_hard = genparts[hard_process]

        
        higgs = genparts_hard[genparts_hard.pdgId == 25]
        top = genparts_hard[genparts_hard.pdgId == 6]
        antitop = genparts_hard[genparts_hard.pdgId == -6]
        
        # I want to take hardProcess, final state, BEFORE FSR particles, which have higgs, top, antitop as parents
        # These will become the hard particles 
        from_higgs = genparts_hard.parent.pdgId == 25
        from_top = genparts_hard.parent.pdgId == 6
        from_antitop = genparts_hard.parent.pdgId == -6

        genparts_initial = genparts[initial]

        isr = genparts_initial[(genparts_initial.hasFlags(['fromHardProcess','isPrompt','isHardProcess'])) &
                                  (genparts_initial.status == 23) & (genparts_initial.pdgId != 25)& (abs(genparts_initial.pdgId) != 6)]
        has_isr = ak.num(isr)!=0
        isr_idx = ak.flatten(ak.fill_none(ak.pad_none(local_index_all[initial][(genparts_initial.hasFlags(['fromHardProcess','isPrompt','isHardProcess'])) &
                                  (genparts_initial.status == 23) & (genparts_initial.pdgId != 25)& (abs(genparts_initial.pdgId) != 6)], 1), 0))

        ######
        # Get the hard process particles
        part_from_top = genparts_hard[from_top]
        W_from_top = ak.flatten(part_from_top[part_from_top.pdgId ==24])
        b_from_top = ak.flatten(part_from_top[part_from_top.pdgId ==5])
        part_from_antitop = genparts_hard[from_antitop]
        W_from_antitop = ak.flatten(part_from_antitop[part_from_antitop.pdgId == -24])
        b_from_antitop = ak.flatten(part_from_antitop[part_from_antitop.pdgId == -5])

        b_from_top_idx = ak.flatten(local_index_all[hard_process][from_top][part_from_top.pdgId ==5])
        W_from_top_idx = ak.flatten(local_index_all[hard_process][from_top][part_from_top.pdgId == 24])
        b_from_antitop_idx = ak.flatten(local_index_all[hard_process][from_antitop][part_from_antitop.pdgId == -5])
        W_from_antitop_idx = ak.flatten(local_index_all[hard_process][from_antitop][part_from_antitop.pdgId == -24])

        # This works because they are the firstCopy of the hard_process particles with higgs as parent. We are skipping all the decay chain of the higgs
        b_from_higgs_idx = local_index_all[hard_process][from_higgs]
        b_from_higgs = genparts_hard[from_higgs]

        #--------------
        # Converting local index to global index, already corrected with the offsets
        b_from_top_idxG = ak.to_numpy(b_from_top_idx + genparts_offsets[:-1], allow_missing=False)
        W_from_top_idxG = ak.to_numpy(W_from_top_idx + genparts_offsets[:-1], allow_missing=False)
        b_from_antitop_idxG = ak.to_numpy(b_from_antitop_idx + genparts_offsets[:-1], allow_missing=False)
        W_from_antitop_idxG = ak.to_numpy(W_from_antitop_idx + genparts_offsets[:-1], allow_missing=False)
        isr_idxG = ak.to_numpy(isr_idx + genparts_offsets[:-1], allow_missing=False)
        b_from_higgs_idxG = ak.to_numpy(b_from_higgs_idx + genparts_offsets[:-1], allow_missing=False)

        # Some inputs needed for numba functions for indexing
        
        genparts_flat_eta = ak.without_parameters(genparts_flat.eta, behavior={})
        genparts_flat_phi = ak.without_parameters(genparts_flat.phi, behavior={})
        genparts_flat_pt = ak.without_parameters(genparts_flat.pt, behavior={})
        genparts_flat_pdgId = ak.without_parameters(genparts_flat.pdgId, behavior={})
        genparts_flat_statusFlags = ak.without_parameters(genparts_flat.statusFlags, behavior={})
        
        firstgenpart_idxG = ak.firsts(genparts[:,0].children).genPartIdxMotherG
        firstgenpart_idxG_numpy = ak.to_numpy( firstgenpart_idxG, allow_missing=False)
        local_ind = ak.to_numpy(ak.local_index(firstgenpart_idxG), allow_missing=False)
        nevents = firstgenpart_idxG_numpy.shape[0]

        ### Analyze the W decay
        W_from_top_islep, W_from_top_decay = analyze_W_flat( W_from_top_idxG, 
                                                             children_idxG_flat,
                                                             genparts_flat_statusFlags,
                                                             genparts_flat_pdgId,
                                                             firstgenpart_idxG_numpy,
                                                             genparts_offsets,
                                                             nevents)
                                                            

        W_from_antitop_islep, W_from_antitop_decay = analyze_W_flat( W_from_antitop_idxG, 
                                                                     children_idxG_flat,
                                                                     genparts_flat_statusFlags,
                                                                     genparts_flat_pdgId,
                                                                     firstgenpart_idxG_numpy,
                                                                     genparts_offsets,
                                                                     nevents)
        
        # assuming semilep only
        W_had_decay_idx = np.where(~W_from_top_islep[:,None],W_from_top_decay, W_from_antitop_decay )
        W_lep_decay_idx = np.where(W_from_top_islep[:,None], W_from_top_decay, W_from_antitop_decay )

        # Now getting all the global Idx of particles for which we want to analyze the decays chain,
        # looking for the highest pt emission after FSR radiation
        part_input_G  = np.concatenate([b_from_top_idxG[:,None],
                                        b_from_antitop_idxG[:,None],
                                        b_from_higgs_idxG,
                                        isr_idxG[:,None],
                                        W_had_decay_idx,
                                        ], axis=1)

        parton_decay_id = analyze_parton_decays_flat_nomesons(part_input_G,
                                                              children_idxG_flat,
                                                              genparts_flat_eta,
                                                              genparts_flat_phi,
                                                              genparts_flat_pt,
                                                              genparts_flat_pdgId,
                                                              self.dr_min_postfsr,
                                                              firstgenpart_idxG_numpy,
                                                              genparts_offsets,
                                                              nevents)

        
        b_from_top_lastcopy = genparts_flat[parton_decay_id[:,0]]
        b_from_antitop_lastcopy = genparts_flat[parton_decay_id[:,1]]
        b_from_higgs_lastcopy = genparts_flat[parton_decay_id[:,2:4]]
        isr_lastcopy = genparts_flat[parton_decay_id[:,4]]
        part_from_Whad_lastcopy = genparts_flat[parton_decay_id[:,5]]
        # We don't take the last copy of the leptonic particles, but the born one
        part_from_Wlep = genparts_flat[W_lep_decay_idx]
        
        # Now we can perform the genmatching
        quarks_initial = genparts_flat[part_input_G]
        quarks_lastcopy = genparts_flat[parton_decay_id]

        quarks_provenance = np.zeros(parton_decay_id.shape, dtype=np.int32)
        quarks_provenance[:, 0] = np.where(W_from_top_islep, 3, 2)  #b from leptonic top =3, hadronic =2
        quarks_provenance[:, 1] = np.where(W_from_antitop_islep, 3, 2)
        quarks_provenance[:, 2:4] = 1 #Higgs
        quarks_provenance[:, 4] = 4 #isr
        quarks_provenance[:, 5:7] = 5 #W hadronic decay

        # Assign provenance
        # 1 - from higgs
        # 2 - from top hadronic bquark
        # 3 - from top leptonic bquark
        # 4 - from ISR
        # 5 - from W hadronic decay
        quarks_initial["provenance"] = ak.Array(quarks_provenance, behavior={})
        quarks_lastcopy["provenance"] = quarks_initial["provenance"]
        

        # Calling our general object_matching function.
        # The output is an awkward array with the shape of the second argument and None where there is no matching.
        # So, calling like this, we will get out an array of matched_quarks with the dimension of the JetGood.
        matched_quarks, matched_jets, deltaR_matched = object_matching(
            quarks_lastcopy, self.events.JetGood, dr_min=self.dr_min
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
        self.matched_partons_mask = ~ak.is_none(self.events.JetGoodMatched, axis=1)

        self.events["LeptonGenLevel"] = part_from_Wlep
        self.events["HiggsGen"] = higgs
        
        self.events["TopGen"] = top
        self.events["AntiTopGen"] = antitop
        self.events["TopGen_islep"] = W_from_top_islep
        self.events["AntiTopGen_islep"] = W_from_antitop_islep
        

    def count_partons(self):
        self.events["nPartonLastCopyMatched"] = ak.count(
            self.events.PartonLastCopyMatched.pt, axis=1
        )  # use count since we have None

    def process_extra_after_presel(self, variation) -> ak.Array:

        if self._sample == "ttHTobb":
            self.do_parton_matching_ttHbb()
        self.count_partons()
