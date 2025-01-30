import spox.opset.ai.onnx.v17 as op
from spox import argument, build, inline, Tensor
import tensorflow as tf
import os
import tf2onnx
import onnx
import onnxruntime as rt
import numpy as np

columns = [
    "era",
    "higgs1_reco_pt",
    "higgs1_reco_eta",
    "higgs1_reco_phi",
    "higgs1_reco_mass",
    "higgs2_reco_pt",
    "higgs2_reco_eta",
    "higgs2_reco_phi",
    "higgs2_reco_mass",
    "HT",
    "higgs1_DeltaRjj",
    "higgs2_DeltaRjj",
    "minDeltaR_Higgjj",
    "maxDeltaR_Higgjj",
    "higgs1_helicityCosTheta",
    "higgs2_helicityCosTheta",
    "hh_CosThetaStar_CS",
    "hh_vec_mass",
    "hh_vec_pt",
    "hh_vec_eta",
    "hh_vec_DeltaR",
    "hh_vec_DeltaPhi",
    "hh_vec_DeltaEta",
    "higgs1_reco_jet1_pt",
    "higgs1_reco_jet1_eta",
    "higgs1_reco_jet1_phi",
    "higgs1_reco_jet1_mass",
    "higgs1_reco_jet2_pt",
    "higgs1_reco_jet2_eta",
    "higgs1_reco_jet2_phi",
    "higgs1_reco_jet2_mass",
    "higgs2_reco_jet1_pt",
    "higgs2_reco_jet1_eta",
    "higgs2_reco_jet1_phi",
    "higgs2_reco_jet1_mass",
    "higgs2_reco_jet2_pt",
    "higgs2_reco_jet2_eta",
    "higgs2_reco_jet2_phi",
    "higgs2_reco_jet2_mass",
    "add_jet1pt_pt",
    "add_jet1pt_eta",
    "add_jet1pt_phi",
    "add_jet1pt_mass",
    "sigma_over_higgs1_reco_mass",
    "sigma_over_higgs2_reco_mass",
]

if __name__ == "__main__":
    main_dir = "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_morphing"

    h5_files = [x for x in os.listdir(main_dir) if x.endswith(".keras")]

    print(f"Processing {h5_files[0]}")

    tot_len = 1

    model = tf.keras.models.load_model(os.path.join(main_dir, h5_files[0]))
    model_ratio = tf.keras.models.Model(
        inputs=model.input, outputs=model.output[:, 1] / model.output[:, 0]
    )

    onnx_model_ratio_sum, _ = tf2onnx.convert.from_keras(
        model_ratio,
        input_signature=[tf.TensorSpec(shape=(None, len(columns)), dtype=tf.float32)],
    )

    b = argument(Tensor(np.float32, ("N", len(columns))))

    for h5_file in h5_files[1:]:
        tot_len += 1
        print(f"\n\nAdding {h5_file}")
        model_add = tf.keras.models.load_model(os.path.join(main_dir, h5_file))
        model_ratio_add = tf.keras.models.Model(
            inputs=model_add.input,
            outputs=model_add.output[:, 1] / model_add.output[:, 0],
        )

        onnx_model_ratio_add, _ = tf2onnx.convert.from_keras(
            model_ratio_add,
            input_signature=[
                tf.TensorSpec(shape=(None, len(columns)), dtype=tf.float32)
            ],
        )

        print(b)
        (r,) = inline(onnx_model_ratio_sum)(b).values()
        (r1,) = inline(onnx_model_ratio_add)(b).values()
        print(r)
        print(r1)

        s = op.add(r, r1)

        onnx_model_ratio_sum = build({"args_0": b}, {"sum_w": s})

    print(f"\ntotal length: {tot_len}")
    (r_sum,) = inline(onnx_model_ratio_sum)(b).values()
    a = op.div(r_sum, op.constant(value_float=tot_len))

    onnx_model_ratio_avg = build({"args_0": b}, {"avg_w": a})
    onnx_model_name = f"{main_dir}/average_model_from_keras.onnx"
    if os.path.exists(onnx_model_name):
        print(f"Removing {onnx_model_name}")
        os.remove(onnx_model_name)
    onnx.save(onnx_model_ratio_avg, onnx_model_name)
    print(f"Model saved as {onnx_model_name}")
