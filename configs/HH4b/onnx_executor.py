from HH4b_parton_matching_config import SPANET_MODEL

from utils.onnx_executor_common import OnnxExecutorFactory

onnx_model_dict={
    "SPANET": SPANET_MODEL,
}

def get_executor_factory(executor_name, **kwargs):
    return OnnxExecutorFactory(onnx_model_dict=onnx_model_dict, **kwargs)