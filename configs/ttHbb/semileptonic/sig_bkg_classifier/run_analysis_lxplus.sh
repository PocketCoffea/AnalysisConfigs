BASE_FOLDER=/afs/cern.ch/work/m/mmarcheg/AnalysisConfigs/configs/ttHbb/semileptonic/sig_bkg_classifier

pocket-coffea run --cfg $BASE_FOLDER/config_ntuples_exporter_dctr.py -ro $BASE_FOLDER/params/run_options_lxplus.yaml -o ntuples_dctr_sfcalibrated -e dask@lxplus --process-separately
