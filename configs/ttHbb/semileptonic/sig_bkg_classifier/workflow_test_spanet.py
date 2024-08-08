import awkward as ak
from pocket_coffea.workflows.tthbb_base_processor import ttHbbBaseProcessor
from dask.distributed import get_worker

import numpy as np

class TestDaskProcessor(ttHbbBaseProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)


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
                f"/work/dvalsecc/ttHbb/AnalysisConfigs/configs/ttHbb/semileptonic/sig_bkg_classifier/spanet.onnx", 
                sess_options = sess_options,
                providers=['CPUExecutionProvider']
            )
        else:
            model_session = worker.data['model_session']

        print(model_session)


        jets_padded = ak.fill_none(ak.pad_none(self.events.Jet, 16, clip=True), {"btag":0., "pt":0., "phi":0., "eta":0.})

        data = np.transpose(
            np.stack([
                ak.to_numpy(jets_padded.eta),  #WRONG --> just for testing
                ak.to_numpy(jets_padded.eta),
                ak.to_numpy(jets_padded.eta),
                ak.to_numpy(jets_padded.eta),
                ak.to_numpy(jets_padded.phi),
            ak.to_numpy(jets_padded.phi),
                ak.to_numpy(jets_padded.pt),
                
            ]),
            axes=[1,2,0]).astype(np.float32)
        
        mask = ~ak.to_numpy(jets_padded.pt == 0)
        
        met_data = np.stack([ak.to_numpy(self.events.MET.phi),
                             ak.to_numpy(self.events.MET.phi),
                             ak.to_numpy(self.events.MET.phi),
                             #ak.to_numpy(self.events.MET.pt)
                             np.log(1+ ak.to_numpy(self.events.MET.pt))
                             ], axis=1)[:,None,:].astype(np.float32)
        
        lep_data = np.stack([ak.to_numpy(self.events.LeptonGood[:,0].eta),
                             ak.to_numpy(self.events.LeptonGood[:,0].phi),
                             ak.to_numpy(self.events.LeptonGood[:,0].phi),
                             ak.to_numpy(self.events.LeptonGood[:,0].phi),
                             #ak.to_numpy(self.events.LeptonGood[:,0].pt)
                             np.log(1+ ak.to_numpy(self.events.LeptonGood[:,0].pt))
                             ], axis=1)[:,None,:].astype(np.float32)
        
        ht_array = np.sum(ak.to_numpy(jets_padded.pt), axis=1)[:,None, None].astype(np.float32)
        
        mask_global = np.ones(shape=[met_data.shape[0], 1]) == 1
        
        njets_good = ak.sum(mask, axis=1)

        outputs = model_session.run(input_feed={
        "Jet_data": data,
        "Jet_mask": mask,
        "Met_data": met_data,
        "Met_mask": mask_global,
        "Lepton_data": lep_data,
        "Lepton_mask": mask_global,
        "Event_data": ht_array, 
        "Event_mask": mask_global},
        output_names=["t1_assignment_probability", "t2_assignment_probability",
                     "h_assignment_probability",
                     "EVENT/tthbb","EVENT/ttbb", "EVENT/ttlf"]
        )

        print(outputs)
