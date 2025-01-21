import sys
from pocket_coffea.executors.executors_T3_CH_PSI import DaskExecutorFactory

from VBF_HH4b_test_config import SPANET_MODEL, DNN_MODEL

sys.path.append("../../")
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
        inference_session_plugin_DNN = WorkerInferenceSessionPlugin(DNN_MODEL, "DNN")
        # Registering the session plugin
        self.dask_client.register_worker_plugin(inference_session_plugin_spanet)
        self.dask_client.register_worker_plugin(inference_session_plugin_DNN)


def get_executor_factory(executor_name, **kwargs):
    return OnnxExecutorFactory(**kwargs)
