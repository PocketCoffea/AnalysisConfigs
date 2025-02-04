from VBF_HH4b_config import SPANET_MODEL, VBF_GGF_DNN_MODEL, BKG_MORPHING_DNN_MODEL

from utils.onnx_executor_common import OnnxExecutorFactory

onnx_model_dict={
    "SPANET": SPANET_MODEL,
    "VBF_GGF_DNN": VBF_GGF_DNN_MODEL,
    "BKG_MORPHING_DNN": BKG_MORPHING_DNN_MODEL,
}


def get_executor_factory(executor_name, **kwargs):
    return OnnxExecutorFactory(onnx_model_dict=onnx_model_dict, **kwargs)