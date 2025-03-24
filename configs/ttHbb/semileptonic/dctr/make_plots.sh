for YEAR in 2016_PreVFP 2016_PostVFP 2017 2018; do
    pocket-coffea make-plots -j 4 -oy $YEAR -o plots_$YEAR --overwrite
done
