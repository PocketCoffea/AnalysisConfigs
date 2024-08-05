from pocket_coffea.executors.executors_T3_CH_PSI import DaskExecutorFactory
from dask.distributed import WorkerPlugin, Worker, Client


class WorkerInferenceSessionPlugin(WorkerPlugin):    
    async def setup(self, worker: Worker):
        import onnxruntime as ort

        sess_options = ort.SessionOptions()

        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            "/pnfs/psi.ch/cms/trivcat/store/user/mmarcheg/ttHbb/multiclassifier_full_Run2_btag_LMH_8M_balance_events/spanet_output/version_3/spanet.onnx", 
            sess_options = sess_options,
            providers=['CPUExecutionProvider']
        )

        worker.data["model_session"] = session

        
# Create an instance of the plugin
inference_session_plugin = WorkerInferenceSessionPlugin()

class OnnxExecutorFactory(DaskExecutorFactory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        super().setup()
        # now setting up and registering the ONNX plugin
        inference_session_plugin = WorkerInferenceSessionPlugin()
        # Registering the session plugin
        self.dask_client.register_worker_plugin(inference_session_plugin)






def get_executor_factory(executor_name, **kwargs):
    return OnnxExecutorFactory(**kwargs)
