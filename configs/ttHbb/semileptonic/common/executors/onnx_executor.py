import os
from pocket_coffea.executors.executors_T3_CH_PSI import DaskExecutorFactory
from dask.distributed import WorkerPlugin, Worker, Client


class WorkerInferenceSessionPlugin(WorkerPlugin):    
    def __init__(self, model_path, session_name):
        super().__init__()
        self.model_path = model_path
        self.session_name = session_name
        if self.model_path is None:
            raise ValueError(f"{self.session_name}: No path to the ONNX model specified.")

    async def setup(self, worker: Worker):
        if os.path.exists("/afs/cern.ch/work"):
            import sys
            sys.path.append("/afs/cern.ch/work/m/mmarcheg/ttHbb/envs/configs/lib/python3.11/site-packages/")
        import onnxruntime as ort

        sess_options = ort.SessionOptions()

        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        
        session = ort.InferenceSession(
            self.model_path,
            sess_options = sess_options,
            providers=['CPUExecutionProvider']
        )

        worker.data[f"model_session_{self.session_name}"] = session

        
# Create an instance of the plugin
#inference_session_plugin = WorkerInferenceSessionPlugin()

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
