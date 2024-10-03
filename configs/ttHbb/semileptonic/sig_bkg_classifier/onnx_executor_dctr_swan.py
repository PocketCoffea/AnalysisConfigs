from pocket_coffea.executors.executors_cern_swan import DaskExecutorFactory
import onnx_executor
from onnx_executor import WorkerInferenceSessionPlugin
from distributed.diagnostics.plugin import UploadFile
import importlib.util
from dask.distributed import WorkerPlugin

class PackageChecker(WorkerPlugin):
    def __init__(self, package_name):
        self.package_name = package_name

    def setup(self, worker):
        try:
            spec = importlib.util.find_spec(self.package_name)
            if spec is None:
                raise ImportError(f"Package {self.package_name} not found")
        except ImportError:
            worker.log.error(f"Package {self.package_name} not found. Restarting worker...")
            worker.close()
            raise SystemExit(1)


class OnnxExecutorFactory(DaskExecutorFactory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        super().setup()
        self.dask_client.register_worker_plugin(PackageChecker("pocket_coffea"))
        # now setting up and registering the ONNX plugin
        self.spanet_model_path = self.run_options.get("spanet_model", None)
        self.dctr_model_path = self.run_options.get("dctr_model", None)
        self.dask_client.register_worker_plugin(UploadFile(self.spanet_model_path))
        self.dask_client.register_worker_plugin(UploadFile(self.dctr_model_path))
        
        inference_session_plugin = WorkerInferenceSessionPlugin(self.spanet_model_path.split("/")[-1], "spanet")
        inference_session_plugin_dctr = WorkerInferenceSessionPlugin(self.dctr_model_path.split("/")[-1], "dctr")
        # Registering the session plugin
        self.dask_client.register_worker_plugin(inference_session_plugin)
        self.dask_client.register_worker_plugin(inference_session_plugin_dctr)

def get_executor_factory(executor_name, **kwargs):
    return OnnxExecutorFactory(**kwargs)
