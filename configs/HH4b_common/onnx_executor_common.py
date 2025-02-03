from pocket_coffea.executors.executors_T3_CH_PSI import DaskExecutorFactory

from .inference_session_onnx import WorkerInferenceSessionPlugin


class OnnxExecutorFactory(DaskExecutorFactory):

    def __init__(self, onnx_model_dict, **kwargs):
        self.onnx_model_dict = onnx_model_dict
        super().__init__(**kwargs)

    def setup(self):
        super().setup()
        # now setting up and registering the ONNX plugin

        for model_name, model in self.onnx_model_dict.items():
            if model:
                inference_session_plugin = WorkerInferenceSessionPlugin(
                    model, model_name
                )
                self.dask_client.register_worker_plugin(inference_session_plugin)
