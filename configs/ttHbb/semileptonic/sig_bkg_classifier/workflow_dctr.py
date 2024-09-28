import numpy as np
import awkward as ak
from dask.distributed import get_worker
import workflow_spanet
from workflow_control_regions import ControlRegionsProcessor
import quantile_transformer
from quantile_transformer import WeightedQuantileTransformer

def get_input_features(events, mask=None, only=None):
    input_features = {
        "njet" : ak.num(events.JetGood),
        "nbjet" : ak.num(events.BJetGood),
        "ht" : events.events.ht,
        "ht_b" : events.events.bjets_ht,
        "ht_light" : events.events.lightjets_ht,
        "drbb_avg" : events.events.drbb_avg,
        "mbb_max" : events.events.mbb_max,
        "mbb_min" : events.events.mbb_min,
        "mbb_closest" : events.events.mbb_closest,
        "drbb_min" : events.events.drbb_min,
        "detabb_min" : events.events.detabb_min,
        "dphibb_min" : events.events.dphibb_min,
        "jet_pt_1" : events.JetGood.pt[:,0],
        "jet_pt_2" : events.JetGood.pt[:,1],
        "jet_pt_3" : events.JetGood.pt[:,2],
        "jet_pt_4" : events.JetGood.pt[:,3],
        "bjet_pt_1" : events.BJetGood.pt[:,0],
        "bjet_pt_2" : events.BJetGood.pt[:,1],
        "bjet_pt_3" : events.BJetGood.pt[:,2],
        "jet_eta_1" : events.JetGood.eta[:,0],
        "jet_eta_2" : events.JetGood.eta[:,1],
        "jet_eta_3" : events.JetGood.eta[:,2],
        "jet_eta_4" : events.JetGood.eta[:,3],
        "bjet_eta_1" : events.BJetGood.eta[:,0],
        "bjet_eta_2" : events.BJetGood.eta[:,1],
        "bjet_eta_3" : events.BJetGood.eta[:,2],
    }
    for key in input_features.keys():
        input_features[key] = ak.to_numpy(input_features[key])
    if only is not None:
        input_features = {k : v for k, v in input_features.items() if k in only}
    if mask is not None:
        for key in input_features.keys():
            input_features[key] = input_features[key][mask]

    return input_features

class DCTRInferenceProcessor(ControlRegionsProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        if not "dctr_model" in self.workflow_options:
            raise ValueError("Key `dctr_model` not found in workflow options. Please specify the path to the ONNX model.")
        elif not self.workflow_options["dctr_model"].endswith(".onnx"):
            raise ValueError("Key `dctr_model` should be the path of an ONNX model.")

    def process_extra_after_presel(self, variation) -> ak.Array:
        super().process_extra_after_presel(variation)

        try:
            worker = get_worker()
        except ValueError:
            worker = None

        if worker is None:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            model_session = ort.InferenceSession(
                self.workflow_options["dctr_model"],
                sess_options = sess_options,
                providers=['CPUExecutionProvider']
            )
        else:
            model_session = worker.data['model_session_dctr']

        print(model_session)

        input_features = get_input_features(self.events)
        data = np.stack(list(input_features.values()), axis=1).astype(np.float32)

        out = model_session.run(output_names='output', input_feed={'input': data})[0]
        dctr_weight = out / (1 - out)

        dctr_dict = {
            "score": out,
            "weight":
        }
