import awkward as ak
import numpy as np
import numba
from numba import njit
from pocket_coffea.lib.parton_provenance import reverse_index_array

@njit
def analyze_parton_from_vbf_quarks(
    vbf_quarks_idx,
    vbf_quarks_pdgId,
    children_idxG_flat,
    genpart_pdgId_flat,
    genpart_offsets,
    genpart_LastCopy_flat,
    genpart_pt_flat,
    nevents,
    firstgenpart_idxG_numpy
):
    # get the children ofthe vbf_quarks which have the same pdgId of the mother iteratively until we reach the last copy

    out = np.zeros(vbf_quarks_idx.shape, dtype="int64")-1

    for iev in range(vbf_quarks_idx.shape[0]):
        for ipart in range(vbf_quarks_idx.shape[1]):
            p_id = vbf_quarks_idx[iev][ipart]
            while not genpart_LastCopy_flat[p_id]:
                children_idxs = reverse_index_array(children_idxG_flat[p_id],
                                    firstgenpart_idxG_numpy,
                                    genpart_offsets, nevents)
                # children_idxs = children_idxG_flat[p_id]

                #get the children with the same pdgId as the mother with highest pt
                max_pt = -1
                max_pt_idx = -1
                if genpart_LastCopy_flat[p_id]:
                    out[iev][ipart] = p_id
                    continue

                for child_idx in children_idxs:
                    if genpart_pdgId_flat[child_idx] != vbf_quarks_pdgId[iev][ipart]:
                        continue
                    child_pt = genpart_pt_flat[child_idx]
                    if child_pt > max_pt:
                        max_pt_idx = child_idx
                        max_pt = child_pt

                if  genpart_LastCopy_flat[max_pt_idx]:
                    out[iev][ipart] = max_pt_idx
                p_id = max_pt_idx
                if out[iev][ipart]!=-1:
                    break
    return out