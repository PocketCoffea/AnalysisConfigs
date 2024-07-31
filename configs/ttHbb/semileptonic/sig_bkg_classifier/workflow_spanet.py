import awkward as ak
import workflow
from workflow import ttbarBackgroundProcessor
from dask.distributed import get_worker

import numpy as np

class SpanetInferenceProcessor(ttbarBackgroundProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        if not "spanet_model" in self.workflow_options:
            raise ValueError("Key `spanet_model` not found in workflow options. Please specify the path to the ONNX model.")
        elif type(self.workflow_options) != str:
            raise ValueError("Key `spanet_model` should be a string.")
        elif not self.workflow_options["spanet_model"].endswith(".onnx"):
            raise ValueError("Key `spanet_model` should be the path of an ONNX model.")


    def process_extra_after_presel(self, variation) -> ak.Array:

        try:
            worker = get_worker()
        except ValueError:
            worker = None

        if worker is None:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            model_session = ort.InferenceSession(
                self.workflow_options["spanet_model"],
                sess_options = sess_options,
                providers=['CPUExecutionProvider']
            )
        else:
            model_session = worker.data['model_session']

        print(model_session)

        btagging_algorithm = self.events.JetGood[self.params.btagging.working_point[self._year]["btagging_algorithm"]]
        jets_padded = ak.fill_none(ak.pad_none(self.events.JetGood, 16, clip=True),
                                   {btagging_algorithm:0., "btag_L":0, "btag_M":0, "btag_H":0,"pt":0., "phi":0., "eta":0.}
                                   )

        data = np.transpose(
            np.stack([
                ak.to_numpy(jets_padded.btag_H),
                ak.to_numpy(jets_padded.btag_M),
                ak.to_numpy(jets_padded.btag_L),
                np.sin(ak.to_numpy(jets_padded.phi)),
                np.cos(ak.to_numpy(jets_padded.phi)),
                ak.to_numpy(jets_padded.eta),
                np.log(1 + ak.to_numpy(jets_padded.pt))
            ]),
            axes=[1,2,0]).astype(np.float32)
        
        mask = ~ak.to_numpy(jets_padded.pt == 0)
        
        met_data = np.stack([np.sin(ak.to_numpy(self.events.MET.phi)),
                             np.cos(ak.to_numpy(self.events.MET.phi)),
                             ak.to_numpy(self.events.MET.eta),
                             np.log(1+ ak.to_numpy(self.events.MET.pt))
                             ], axis=1)[:,None,:].astype(np.float32)
        
        lep_data = np.stack([ak.to_numpy(self.events.LeptonGood[:,0].is_electron).astype(np.int32),
                             np.sin(ak.to_numpy(self.events.LeptonGood[:,0].phi)),
                             np.cos(ak.to_numpy(self.events.LeptonGood[:,0].phi)),
                             ak.to_numpy(self.events.LeptonGood[:,0].eta),
                             np.log(1 + ak.to_numpy(self.events.LeptonGood[:,0].pt))
                             ], axis=1)[:,None,:].astype(np.float32)
        
        ht_array = self.events.JetGood_Ht[:,None, None].astype(np.float32)
        
        mask_global = np.ones(shape=[met_data.shape[0], 1]) == 1

        outputs = model_session.run(input_feed={
            "Jet_data": data,
            "Jet_mask": mask,
            "Met_data": met_data,
            "Met_mask": mask_global,
            "Lepton_data": lep_data,
            "Lepton_mask": mask_global,
            "Event_data": ht_array, 
            "Event_mask": mask_global},
        output_names=[
                     "EVENT/tthbb", "EVENT/ttbb", "EVENT/ttcc", "EVENT/ttlf"]
        )

        print(outputs)
