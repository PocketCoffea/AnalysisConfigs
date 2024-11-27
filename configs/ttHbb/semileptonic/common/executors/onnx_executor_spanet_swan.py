from pocket_coffea.executors.executors_cern_swan import DaskExecutorFactory
import onnx_executor
from onnx_executor import WorkerInferenceSessionPlugin

class OnnxExecutorFactory(DaskExecutorFactory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        super().setup()
        # now setting up and registering the ONNX plugin
        self.spanet_model_path = self.run_options.get("spanet_model", None)
        inference_session_plugin = WorkerInferenceSessionPlugin(self.spanet_model_path, "spanet")
        # Registering the session plugin
        self.dask_client.register_worker_plugin(inference_session_plugin)

def get_executor_factory(executor_name, **kwargs):
    return OnnxExecutorFactory(**kwargs)
