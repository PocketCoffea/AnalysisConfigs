BASE_FOLDER=/work/mmarcheg/AnalysisConfigs/configs/ttHbb/semileptonic/sig_bkg_classifier

pocket-coffea run --cfg $BASE_FOLDER/config_control_regions.py -ro $BASE_FOLDER/params/run_options.yaml -o control_regions_without_shape_variations --executor-custom-setup $BASE_FOLDER/onnx_executor.py --process-separately
