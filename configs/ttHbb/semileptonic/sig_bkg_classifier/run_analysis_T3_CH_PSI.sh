BASE_FOLDER=/work/mmarcheg/AnalysisConfigs/configs/ttHbb/semileptonic/sig_bkg_classifier

pocket-coffea run --cfg $BASE_FOLDER/config_ntuples_exporter_dctr_spanet_inference.py -ro $BASE_FOLDER/params/run_options_short.yaml -o ntuples_dctr_sfcalibrated --executor-custom-setup $BASE_FOLDER/onnx_executor.py --process-separately
