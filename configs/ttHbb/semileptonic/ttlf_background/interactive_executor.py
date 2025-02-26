from coffea import processor as coffea_processor
from dask.distributed import Client
from pocket_coffea.executors.executors_T3_CH_PSI import DaskExecutorFactory
            

class InteractiveExecutorFactory(DaskExecutorFactory):

    def __init__(self, run_options, outputdir, **kwargs):
        if "sched-url" not in run_options or run_options["sched-url"] is None:
            raise Exception("`sched-url` key not provided in the custom run options! Please provide the URL of the Dask scheduler!")
        self.sched_url = run_options["sched-url"]
        super().__init__(run_options, outputdir, **kwargs)
        
    def setup(self):
        ''' Start the DASK cluster here'''
        # At INFN AF, the best way to handle DASK clusters is to create them via the Dask labextension and then connect the client to it in your code
        self.dask_client = Client(address=self.sched_url)
        
    def customized_args(self):
        args = super().customized_args()
        args["client"] = self.dask_client
        args["treereduction"] = self.run_options["tree-reduction"]
        args["retries"] = self.run_options["retries"]
        return args
    
    def get(self):
        return coffea_processor.dask_executor(**self.customized_args())

    def close(self):
        self.dask_client.close()

def get_executor_factory(executor_name, **kwargs):
        return InteractiveExecutorFactory(**kwargs)