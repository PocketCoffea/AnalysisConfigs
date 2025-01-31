from dask.distributed import WorkerPlugin, Worker, Client, get_worker

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

        model_session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        worker.data[f"model_session_{self.session_name}"] = model_session
        worker.data[f"input_name_{self.session_name}"] = [input.name for input in model_session.get_inputs()]
        worker.data[f"output_name_{self.session_name}"] = [output.name for output in model_session.get_outputs()]


def get_model_session(model_path, model_session_name):
    try:
        worker = get_worker()
    except ValueError:
        worker = None

    if worker is None:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        model_session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        input_name = [input.name for input in model_session.get_inputs()]
        output_name = [output.name for output in model_session.get_outputs()]

    else:
        model_session = worker.data[f"model_session_{model_session_name}"]
        input_name = worker.data[f"input_name_{model_session_name}"]
        output_name = worker.data[f"output_name_{model_session_name}"]


    return model_session, input_name, output_name