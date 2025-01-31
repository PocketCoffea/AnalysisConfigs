import awkward as ak
import numpy as np
from pocket_coffea.lib.weights_manager import WeightCustom



def getSMEFTweight(i: int):
    """
    Get the weight for the i-th SMEFT parameter point
    """
    name = f"SMEFT_weight_{i}"
    return WeightCustom(
        name = name,
        function = lambda params, events,size,metadata, shape_variation: [ (name,  events.LHEReweightingWeight[:,i])]
    )

