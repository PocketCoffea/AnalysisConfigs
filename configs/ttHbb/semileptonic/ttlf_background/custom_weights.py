from pocket_coffea.lib.weights.weights import WeightLambda
import numpy as np
import awkward as ak

samples_top = ["TTbbSemiLeptonic", "TTToSemiLeptonic", "TTTo2L2Nu"]

SF_top_pt = WeightLambda.wrap_func(
    name="sf_top_pt",
    function=lambda params, metadata, events, size, shape_variations:
            get_sf_top_pt(events, metadata),
    has_variations=False
    )

def get_sf_top_pt(events, metadata):
    if metadata["sample"] in samples_top:
        #print("Computing top pt reweighting for sample: ", metadata["sample"])
        part = events.GenPart
        part = part[~ak.is_none(part.parent, axis=1)]
        part = part[part.hasFlags("isLastCopy")]
        part = part[abs(part.pdgId) == 6]
        part = part[ak.argsort(part.pdgId, ascending=False)]

        arg = {
            "a": 0.103,
            "b": -0.0118, 
            "c": -0.000134,
            "d": 0.973
        }
        top_weight = arg["a"] * np.exp(arg["b"] * part.pt[:,0]) + arg["c"] * part.pt[:,0] + arg["d"]
        antitop_weight = arg["a"] * np.exp(arg["b"] * part.pt[:,1]) + arg["c"] * part.pt[:,1] + arg["d"]
        weight = np.sqrt(ak.prod([top_weight, antitop_weight], axis=0))
        # for i in range(10):
            # print("Top pt: {},   Top SF: {},   AntiTop pt :  {},   AntiTop SF: {}".format(part.pt[i,0], top_weight[i], part.pt[i,1], antitop_weight[i]))
        return weight#, np.zeros(np.shape(weight)), ak.copy(weight)
    else:
        return np.ones(len(events), dtype=np.float64)
