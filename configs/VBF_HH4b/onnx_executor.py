import sys
import os
from pocket_coffea.executors.executors_T3_CH_PSI import DaskExecutorFactory

from VBF_HH4b_config import SPANET_MODEL, VBF_GGF_DNN_MODEL, BKG_MORPHING_DNN_MODEL

# localdir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(f"{localdir}/../../")
# sys.path.append(f"/t3home/mmalucch/AnalysisConfigs/")
# print(sys.path)
from inference_session_onnx import WorkerInferenceSessionPlugin


class OnnxExecutorFactory(DaskExecutorFactory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        super().setup()
        # now setting up and registering the ONNX plugin

        for model, model_name in zip(
            [SPANET_MODEL, VBF_GGF_DNN_MODEL, BKG_MORPHING_DNN_MODEL],
            ["SPANET", "VBF_GGF_DNN", "BKG_MORPHING_DNN"],
        ):
            if model:
                inference_session_plugin = WorkerInferenceSessionPlugin(model, model_name)
                self.dask_client.register_worker_plugin(inference_session_plugin)


def get_executor_factory(executor_name, **kwargs):
    return OnnxExecutorFactory(**kwargs)
