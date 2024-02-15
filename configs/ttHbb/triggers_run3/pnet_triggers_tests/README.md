# Run3 trigger quick study

To run the workflow on lxplus:

```
apptainer shell --bind /afs -B /cvmfs/cms.cern.ch --bind /tmp  --bind /eos/cms/ -B /eos/user/m/mstamenk/ --env KRB5CCNAME=$KRB5CCNAME --bind /etc/sysconfig/ngbauth-submit   /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-cc7-latest


runner.py --cfg config.py -o output1 -e futures -s 6 
```
