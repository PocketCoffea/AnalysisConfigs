import awkward as ak

def mask_efficiency(mask, bool_flatten):
    sum = 0
    if bool_flatten: mask = ak.flatten(mask)
    for i in range(len(mask)):
        if mask[i]:
            sum += 1
    return sum/len(mask)


