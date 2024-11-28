import awkward as ak
import numpy as np
import numba
from numba import njit
from pocket_coffea.lib.parton_provenance import reverse_index_array


@njit
def get_parton_last_copy(
    quarks_idx,
    quarks_pdgId,
    children_idxG_flat,
    genpart_pdgId_flat,
    genpart_offsets,
    genpart_LastCopy_flat,
    genpart_pt_flat,
    nevents,
    firstgenpart_idxG_numpy,
):
    """
    Get the last copy of quarks in the event.

    This function iteratively finds the last copy of quarks by following their children
    until the last copy is reached. The last copy is defined as the particle with the same
    PDG ID as the mother and no further children or marked as the last copy.
    If multiple children have the same PDG ID as the mother, the one with the highest
    pT is chose.

    Parameters:
    quarks_idx (numpy.ndarray): Indices of quarks in the event.
    quarks_pdgId (numpy.ndarray): PDG IDs of the quarks.
    children_idxG_flat (numpy.ndarray): Flattened array of children indices.
    genpart_pdgId_flat (numpy.ndarray): Flattened array of PDG IDs of generated particles.
    genpart_offsets (numpy.ndarray): Offsets for the generated particles.
    genpart_LastCopy_flat (numpy.ndarray): Flattened array indicating if a particle is the last copy.
    genpart_pt_flat (numpy.ndarray): Flattened array of transverse momentum of generated particles.
    nevents (int): Number of events.
    firstgenpart_idxG_numpy (numpy.ndarray): Indices of the first generated particles.

    Returns:
    numpy.ndarray: Indices of the last copy of quarks.
    """

    out = np.zeros(quarks_idx.shape, dtype="int64") - 1

    for iev in range(quarks_idx.shape[0]):
        for ipart in range(quarks_idx.shape[1]):
            p_id = quarks_idx[iev][ipart]
            # if the particle is already the last copy or has no children, it is the last copy
            if genpart_LastCopy_flat[p_id] or len(children_idxG_flat[p_id]) == 0:
                out[iev][ipart] = p_id
                continue
            # loop until the last copy is found
            while (not genpart_LastCopy_flat[p_id]) and (
                not len(children_idxG_flat[p_id]) == 0
            ):
                # get the correct children idx
                children_idxs = reverse_index_array(
                    children_idxG_flat[p_id],
                    firstgenpart_idxG_numpy,
                    genpart_offsets,
                    nevents,
                )

                # get the children with the same pdgId as the mother with highest pt
                max_pt = -1
                max_pt_idx = p_id

                # loop over children
                for child_idx in children_idxs:
                    # if PDG ID is not the same as the mother, skip
                    if genpart_pdgId_flat[child_idx] != quarks_pdgId[iev][ipart]:
                        continue
                    child_pt = genpart_pt_flat[child_idx]
                    # get highest pt among children
                    if child_pt > max_pt:
                        max_pt_idx = child_idx
                        max_pt = child_pt

                # if the last copy is found, set the index
                if (
                    genpart_LastCopy_flat[max_pt_idx]
                    or len(children_idxG_flat[max_pt_idx]) == 0
                    or max_pt == -1
                ):
                    out[iev][ipart] = max_pt_idx

                # update the particle id with the children
                p_id = max_pt_idx

                # if last copy is found, break
                if out[iev][ipart] != -1:
                    break
    return out
