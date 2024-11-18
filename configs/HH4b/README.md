# HH4b

execute the following command to generate the coffea file input to SPANet:
```bash
pocket-coffea run --cfg HH4b_parton_matching_config.py -e dask@T3_CH_PSI --custom-run-options params/t3_run_options.yaml -o <out_dir>
```