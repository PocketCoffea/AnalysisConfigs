import awkward as ak
import workflow
from workflow import ttbarBackgroundProcessor
from dask.distributed import get_worker
import quantile_transformer
from quantile_transformer import WeightedQuantileTransformer

import numpy as np

class SpanetInferenceProcessor(ttbarBackgroundProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        if not "spanet_model" in self.workflow_options:
            raise ValueError("Key `spanet_model` not found in workflow options. Please specify the path to the ONNX model.")
        elif not self.workflow_options["spanet_model"].endswith(".onnx"):
            raise ValueError("Key `spanet_model` should be the path of an ONNX model.")


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
            sess_options.intra_op_num_threads = 1
            model_session = ort.InferenceSession(
                self.workflow_options["spanet_model"],
                sess_options = sess_options,
                providers=['CPUExecutionProvider']
            )
        else:
            model_session = worker.data['model_session_spanet']

        print(model_session)

        btagging_algorithm = self.params.btagging.working_point[self._year]["btagging_algorithm"]
        pad_dict = {btagging_algorithm:0., "btag_L":0, "btag_M":0, "btag_H":0,"pt":0., "phi":0., "eta":0.}
        jets_padded = ak.zip(
            {key : ak.fill_none(ak.pad_none(self.events.JetGood[key], 16, clip=True), value) for key, value in pad_dict.items()}
        )

        data = np.transpose(
            np.stack([
                np.log(1 + ak.to_numpy(jets_padded.pt)),
                ak.to_numpy(jets_padded.eta),
                np.sin(ak.to_numpy(jets_padded.phi)),
                np.cos(ak.to_numpy(jets_padded.phi)),
                ak.to_numpy(jets_padded.btag_L),
                ak.to_numpy(jets_padded.btag_M),
                ak.to_numpy(jets_padded.btag_H)
            ]),
            axes=[1,2,0]).astype(np.float32)

        mask = ~ak.to_numpy(jets_padded.pt == 0)

        met_data = np.stack([np.log(1+ ak.to_numpy(self.events.MET.pt)),
                             ak.zeros_like(self.events.MET.pt).to_numpy(),
                             np.sin(ak.to_numpy(self.events.MET.phi)),
                             np.cos(ak.to_numpy(self.events.MET.phi))
                             ], axis=1)[:,None,:].astype(np.float32)

        lep_data = np.stack([np.log(1 + ak.to_numpy(self.events.LeptonGood[:,0].pt)),
                             ak.to_numpy(self.events.LeptonGood[:,0].eta),
                             np.sin(ak.to_numpy(self.events.LeptonGood[:,0].phi)),
                             np.cos(ak.to_numpy(self.events.LeptonGood[:,0].phi)),
                             ak.to_numpy(self.events.LeptonGood[:,0].is_electron).astype(np.int32),
                             ], axis=1)[:,None,:].astype(np.float32)

        ht_array = ak.to_numpy(self.events.JetGood_Ht[:,None, None]).astype(np.float32)

        mask_global = np.ones(shape=[met_data.shape[0], 1]) == 1

        output_names = ["EVENT/tthbb", "EVENT/ttbb", "EVENT/ttcc", "EVENT/ttlf"]
        outputs = model_session.run(input_feed={
            "Jet_data": data,
            "Jet_mask": mask,
            "Met_data": met_data,
            "Met_mask": mask_global,
            "Lepton_data": lep_data,
            "Lepton_mask": mask_global,
            "Event_data": ht_array, 
            "Event_mask": mask_global},
        output_names=output_names
        )

        outputs_zipped = dict(zip(output_names, outputs))
        self.events["spanet_output"] = ak.zip(
            {
                key.split("/")[-1]: ak.from_numpy(value[:,1]) for key, value in outputs_zipped.items()
            }
        )

        # Transform ttHbb score with quantile transformation
        params_quantile_transformer = self.params["quantile_transformer"][self.events.metadata["year"]]
        transformer = WeightedQuantileTransformer(n_quantiles=params_quantile_transformer["n_quantiles"], output_distribution=params_quantile_transformer["output_distribution"])
        transformer.load(params_quantile_transformer["file"])
        transformed_score = transformer.transform(self.events.spanet_output.tthbb)
        self.events["spanet_output"] = ak.with_field(self.events["spanet_output"], transformed_score, "tthbb_transformed")
