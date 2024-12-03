import awkward as ak
import numpy as np
from numba import njit

def mask_efficiency(mask, bool_flatten):
    sum = 0
    if bool_flatten: mask = ak.flatten(mask)
    for i in range(len(mask)):
        if mask[i]:
            sum += 1
    return sum/len(mask)


@njit
def get_jets_no_higgs(jets_index_all, jets_from_higgs_idx):
    # jets_no_higgs_idx = np.zeros(jet_offset_no_higgs[-1], dtype="int64")-1
    jets_no_higgs_idx = jets_index_all
    print(len(jets_no_higgs_idx), jets_no_higgs_idx)

    # if jets_from_higgs_idx put -1
    for ijet_higgs in jets_from_higgs_idx:
        jets_no_higgs_idx[ijet_higgs] = -1

    return jets_no_higgs_idx