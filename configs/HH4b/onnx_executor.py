import sys
from pocket_coffea.executors.executors_T3_CH_PSI import DaskExecutorFactory

from HH4b_parton_matching_config import SPANET_MODEL

sys.path.append("../../")
#TODO: import from utils
from inference_session_onnx import WorkerInferenceSessionPlugin

class OnnxExecutorFactory(DaskExecutorFactory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        super().setup()
        # now setting up and registering the ONNX plugin
        inference_session_plugin_spanet = WorkerInferenceSessionPlugin(
            SPANET_MODEL, "spanet"
        )
        # Registering the session plugin
        self.dask_client.register_worker_plugin(inference_session_plugin_spanet)


def get_executor_factory(executor_name, **kwargs):
    return OnnxExecutorFactory(**kwargs)
