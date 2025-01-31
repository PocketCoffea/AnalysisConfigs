import awkward as ak
import numba
import numpy as np
from pocket_coffea.lib.parton_provenance import analyze_W_flat, analyze_parton_decays_flat_nomesons


def do_genmatching(events, dr_max_postfsr=1.0):
    '''This function performed the matching of the genparticles to initial quarks.
    More explanation and tests here: https://github.com/valsdav/ttHbb_jets_partons_studies/blob/master/ImproveTruthMatching_final.ipynb '''
    genparts = events.GenPart

    children_idxG = ak.without_parameters(genparts.childrenIdxG, behavior={})
    children_idxG_flat = ak.flatten(children_idxG, axis=1)
    genparts_flat = ak.flatten(genparts)
    genparts_offsets = np.concatenate([[0],np.cumsum(ak.to_numpy(ak.num(genparts, axis=1), allow_missing=True))])
    local_index_all = ak.local_index(genparts, axis=1)

    # Get the initial partons, first copy
    #initial = genparts.genPartIdxMother == 0
    # genparts_initial = genparts[initial]

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


    # Get the ISR
    # More general ISR matching
    outgoing_mask = (genparts.hasFlags(['fromHardProcess','isPrompt','isHardProcess'])) & (genparts.status == 23)
    outgoing = genparts[outgoing_mask]

    isr = outgoing[outgoing.parent.status==21]
    has_isr = ak.num(isr)!=0
    isr_idx =  ak.flatten(ak.fill_none(ak.pad_none(local_index_all[outgoing_mask][outgoing.parent.status==21], 1),0))

    ######
    # Get the hard process particles
    ######
    part_from_top = genparts_hard[from_top]

    # in some events the w is not saved explicitely but already its decays. in those cases we insert a None in the W_from_top
    W_from_top = ak.flatten(ak.pad_none(part_from_top[part_from_top.pdgId ==24], 1, axis=1))
    W_from_top_exist = ~ak.is_none(W_from_top)
    b_from_top = ak.flatten(part_from_top[part_from_top.pdgId ==5])

    part_from_antitop = genparts_hard[from_antitop]
    # in some events the w is not saved explicitely but already its decays. in those cases we insert a None in the W_from_top
    W_from_antitop = ak.flatten(ak.pad_none(part_from_antitop[part_from_antitop.pdgId ==-24], 1, axis=1))
    W_from_antitop_exist = ~ak.is_none(W_from_antitop)
    b_from_antitop = ak.flatten(part_from_antitop[part_from_antitop.pdgId == -5])

    b_from_top_idx = ak.flatten(local_index_all[hard_process][from_top][part_from_top.pdgId ==5])
    W_from_top_idx = ak.flatten(ak.pad_none(local_index_all[hard_process][from_top][part_from_top.pdgId == 24],1, axis=1))
    b_from_antitop_idx = ak.flatten(local_index_all[hard_process][from_antitop][part_from_antitop.pdgId == -5])
    W_from_antitop_idx = ak.flatten(ak.pad_none(local_index_all[hard_process][from_antitop][part_from_antitop.pdgId == -24],1, axis=1))

    W_from_top_idx = ak.fill_none(W_from_top_idx, 0)
    W_from_antitop_idx = ak.fill_none(W_from_antitop_idx, 0)

    # This works because they are the firstCopy of the hard_process particles with higgs as parent. We are skipping all the decay chain of the higgs
    b_from_higgs_idx = local_index_all[hard_process][from_higgs]
    b_from_higgs = genparts_hard[from_higgs]


    #--------------
    b_from_top_idxG = ak.to_numpy(b_from_top_idx + genparts_offsets[:-1], allow_missing=False)
    W_from_top_idxG = ak.to_numpy(W_from_top_idx + genparts_offsets[:-1], allow_missing=False)
    b_from_antitop_idxG = ak.to_numpy(b_from_antitop_idx + genparts_offsets[:-1], allow_missing=False)
    W_from_antitop_idxG = ak.to_numpy(W_from_antitop_idx + genparts_offsets[:-1], allow_missing=False)
    isr_idxG = ak.to_numpy(isr_idx + genparts_offsets[:-1], allow_missing=False)
    b_from_higgs_idxG = ak.to_numpy(b_from_higgs_idx + genparts_offsets[:-1], allow_missing=False)
    W_from_top_exist = ak.to_numpy(W_from_top_exist)
    W_from_antitop_exist = ak.to_numpy(W_from_antitop_exist)

    # Putting -1 in the index if the W is missing 
    W_from_top_idxG = np.where(W_from_top_exist, W_from_top_idxG, -1)
    W_from_antitop_idxG = np.where(W_from_antitop_exist, W_from_antitop_idxG, -1)


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
    # Remember that is the W does not exist the index 0 is returned
    # We need to mask it later
    part_from_Wtop = genparts_flat[W_from_top_decay]
    part_from_Wantitop = genparts_flat[W_from_antitop_decay]

    # Now we need to account for the case where W is not explicitely save in the NAnoAOD,
    # but directly the decays
    direct_Wdecays_fromtop = part_from_top[(abs(part_from_top.pdgId)!=5) & (abs(part_from_top.pdgId)!=24)]
    direct_Wdecays_fromantitop = part_from_antitop[(abs(part_from_antitop.pdgId)!=5) & (abs(part_from_antitop.pdgId)!=24)]

    direct_W_from_top_islep = ak.sum(abs(direct_Wdecays_fromtop.pdgId), axis=1)>=23
    direct_W_from_top_decay_idx = local_index_all[hard_process][from_top][(abs(part_from_top.pdgId)!=5) & (abs(part_from_top.pdgId)!=24)]
    direct_W_from_top_decay_idxG = direct_W_from_top_decay_idx + genparts_offsets[:-1]

    direct_W_from_antitop_islep = ak.sum(abs(direct_Wdecays_fromantitop.pdgId), axis=1)>=23
    direct_W_from_antitop_decay_idx = local_index_all[hard_process][from_antitop][(abs(part_from_antitop.pdgId)!=5) & (abs(part_from_antitop.pdgId)!=24)]
    direct_W_from_antitop_decay_idxG = direct_W_from_antitop_decay_idx + genparts_offsets[:-1]

    # we need to take into account both if it is leptonic and if it exists
    W_had_decay_idx = np.where((~W_from_top_islep & W_from_top_exist)[:, None], W_from_top_decay, np.zeros((len(W_from_top_decay), 2), dtype=int))
    W_had_decay_idx = np.where((~W_from_antitop_islep & W_from_antitop_exist)[:,None], W_from_antitop_decay, W_had_decay_idx )

    W_lep_decay_idx = np.where((W_from_top_islep & W_from_top_exist)[:,None], W_from_top_decay,  np.zeros((len(W_from_top_decay), 2), dtype=int) )
    W_lep_decay_idx = np.where((W_from_antitop_islep & W_from_antitop_exist)[:,None], W_from_antitop_decay, W_lep_decay_idx)

    # adding back the directW decays

    W_had_decay_idx_direct = np.where(~direct_W_from_top_islep & ~W_from_top_exist, direct_W_from_top_decay_idxG, np.zeros((len(W_from_top_decay), 2), dtype=int) )
    W_had_decay_idx_direct = np.where(~direct_W_from_antitop_islep & ~W_from_antitop_exist, direct_W_from_antitop_decay_idxG, W_had_decay_idx_direct)

    W_lep_decay_idx_direct = np.where(direct_W_from_top_islep & ~W_from_top_exist,  direct_W_from_top_decay_idxG, np.zeros((len(W_from_top_decay), 2), dtype=int))
    W_lep_decay_idx_direct = np.where(direct_W_from_antitop_islep & ~W_from_antitop_exist,direct_W_from_antitop_decay_idxG, W_lep_decay_idx_direct)

    W_had_from_direct = ak.sum(W_had_decay_idx_direct, axis=1)>0
    W_lep_from_direct = ak.sum(W_lep_decay_idx_direct, axis=1)>0

    W_had_decay_idx = ak.to_numpy(np.where(W_had_from_direct, W_had_decay_idx_direct, W_had_decay_idx), allow_missing=False)
    W_lep_decay_idx = ak.to_numpy(np.where(W_lep_from_direct, W_lep_decay_idx_direct, W_lep_decay_idx), allow_missing=False)

    part_from_Whad = genparts_flat[W_had_decay_idx]
    part_from_Wlep = genparts_flat[W_lep_decay_idx]

    # Now we can trace the showering of the particles in Pythia
    #looking for the highest pt emission after FSR radiation

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
                                                          dr_max_postfsr,
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

    return [ higgs, top, antitop, isr,
             quarks_initial, quarks_lastcopy,
             part_from_Wlep, part_from_Whad,
             W_from_top_islep, W_from_antitop_islep,
             b_from_top, b_from_antitop, b_from_higgs ]
