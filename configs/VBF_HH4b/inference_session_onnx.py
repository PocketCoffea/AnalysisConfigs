from dask.distributed import WorkerPlugin, Worker, Client

class WorkerInferenceSessionPlugin(WorkerPlugin):
    def __init__(self, model_path, session_name):
        super().__init__()
        self.model_path = model_path
        self.session_name = session_name

    async def setup(self, worker: Worker):
        import onnxruntime as ort

        sess_options = ort.SessionOptions()

        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 1

        session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        worker.data[f"model_session_{self.session_name}"] = session
