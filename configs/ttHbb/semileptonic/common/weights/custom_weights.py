import yaml
import numpy as np
import awkward as ak
import correctionlib
from pocket_coffea.lib.weights.weights import WeightLambda, WeightWrapper, WeightData

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

def sf_ttlf_calib(params, sample, year, njets, jetsHt):
    '''Correction to tt+LF background computed by correcting tt+LF to data minus the other backgrounds in 2D:
    njets-JetsHT bins. Each year has a different correction stored in the correctionlib format.'''
    assert sample == "TTToSemiLeptonic", "This weight is only for TTToSemiLeptonic sample"
    cset = correctionlib.CorrectionSet.from_file(
        params.ttlf_calibration[year]["file"]
    )
    corr = cset[params.ttlf_calibration[year]["name"]]
    w = corr.evaluate(ak.to_numpy(njets), ak.to_numpy(jetsHt))
    return w

def get_njet_reweighting(events, reweighting_map_njet, mask=None):
    njet = ak.num(events.JetGood)
    w_nj = np.ones(len(events))
    if mask is None:
        mask = np.ones(len(events), dtype=bool)
    for nj in range(4,7):
        mask_nj = (njet == nj)
        w_nj = np.where(mask & mask_nj, reweighting_map_njet[nj], w_nj)
    for nj in range(7,21):
        w_nj = np.where(mask & (njet >= 7), reweighting_map_njet[nj], w_nj)
    return w_nj

DCTR_weight = WeightLambda.wrap_func(
    name="dctr_weight",
    function=lambda params, metadata, events, size, shape_variations:
            events.dctr_output.weight,
    has_variations=False
    )

class SF_ttlf_calib(WeightWrapper):
    name = "sf_ttlf_calib"
    has_variations = False

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.jet_coll = "JetGood"

    def compute(self, events, size, shape_variation):
        jetsHt = ak.sum(events[self.jet_coll].pt, axis=1)
        out = sf_ttlf_calib(self._params,
                            sample=self._metadata["sample"],
                            year=self._metadata["year"],
                            # Assuming n*JetCollection* is defined
                            njets=events[f"n{self.jet_coll}"],
                            jetsHt=jetsHt
                            )
        return WeightData(
            name = self.name,
            nominal = out, #out[0] only if has_variations = True
            #up = out[1],
            #down = out[2]
            )

class SF_njet_reweighting(WeightWrapper):
    '''Correction to tt+bb background computed to match data/MC in the number of jets.
    The corection applied during training of the DCTR model is stored in a yaml file.'''
    name = "sf_njet_reweighting"
    has_variations = False

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        assert metadata["sample"] == "TTbbSemiLeptonic", "This weight is only for TTbbSemiLeptonic sample"
        params_njet_reweighting = params.dctr["njet_reweighting"]
        file, key = params_njet_reweighting["file"], params_njet_reweighting["key"]
        with open(file, "r") as f:
            self.reweighting_map_njet = yaml.safe_load(f)[key]

    def compute(self, events, size, shape_variation):
        out = get_njet_reweighting(events, self.reweighting_map_njet)
        return WeightData(
            name = self.name,
            nominal = out
        )