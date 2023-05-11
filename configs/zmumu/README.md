# Full analysis example - Simple Z->mumu 


A full example of all the steps needed to run a full analysis with PocketCoffea is reported, starting from the creation of the datasets list, the customization of parameters and the production of the final shapes and plots.
As an example, a toy example version of the Drell-Yan analysis targeting the Z->mumu channel is implemented.

The main steps that need to be performed are the following:

* Build the json datasets
* Compute the sum of genweights for Monte Carlo datasets and rebuild json datasets
* Define selections: trigger, skim, object preselection, event preselection and categorization
* Define weights and variations
* Define histograms
* Run the processor
* Make the plots from processor output

# Installation
## Install PocketCoffea

PocketCoffea can be installed from pip or directly from the sources. The latter is needed if you wish to contribute to
the core of the framework code and to get the latest under development features. 

The environment needs to be set differently depending on the cluster where you want to run. Please refer to the detailed
guide [Installation guide](https://pocketcoffea.readthedocs.io/en/latest/installation.html). 

We report here briefly the necessary steps to setup the framework from sources. 

```bash
pythom -m venv myenv
source myenv/bin/activate

git clone https://github.com/PocketCoffea/PocketCoffea.git
cd PocketCoffea
pip install -e .  
#-e installs it in "editable" mode so that the modified files are included dynamically in the package.
```

## Configuration files

The parameters specific to the analysis have to be specified in the configuration file. 
This file contains a pure python dictionary named ``cfg`` that is read and manipulated by the ``Configurator`` module.

A dedicated `repository <https://github.com/PocketCoffea/AnalysisConfigs>`_ is setup to collect the config files from
different analysis. Clone the repository **outside** of the PocketCoffea main folder:

```bash
cd ..

# Now downloading the example configuration
git clone https://github.com/PocketCoffea/AnalysisConfigs.git
cd AnalysisConfigs/configs/zmumu

# Ready to start!
```


A dedicated folder ``zmumu`` under the ``configs`` contains all the config files, the datasets definition json file, the
workflows files and possibly extra files with parameters that are needed for this tutorial analysis:

# Dataset preparation
## Build datasets metadata

The datasets of this analysis include a Drell-Yan Monte Carlo dataset and a ``SingleMuon`` dataset. 
We have to look for the corresponding `DAS <https://cmsweb.cern.ch/das/>`_ keys:

- `/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM`
- `/SingleMuon/Run2018C-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD`

The list of datasets has to be written in a structured dictionary together with the corresponding metadata in a json
file.  This json file is then read by the ``build_dataset.py`` script to produce the actual json datasets that are
passed as input to the Coffea processor. The steps are the following:

1) Create a json file that contains the required datasets, ``dataset_definitions.json``. Each entry of the dictionary
corresponds to a dataset. Datasets include a list of DAS keys, the output json dataset path and the metadata. In
addition a label ``sample`` is specified to assign a the "type" of events contained in the dataset (e.g. group QCD datasets from different
HT bins in a single sample).

The general idea is the following:

* The **dataset** key uniquely identifies a dataset.
* The **sample** key is the one used internally in the framework to categorize the type of the events contained in the
  dataset. 
* The **json_output** key defines the path of the destination json file which will store the list of files.
* Several **files** entries can be defined for each dataset, including a list of DAS names and a dedicated metadata dictionary.
* The **metadata** keys should include:
	* For **Monte Carlo**: ``year``, ``isMC`` and ``xsec``.
	* For **Data**: ``year``, ``isMC``, ``era`` and ``primaryDataset``.

For a more detailed discussion about datasets, samples and their meaning in PocketCoffea,
see **MISSING LINK**.

When the json datasets are built, the metadata parameters are linked to the files list, defining a unique dataset entry
with the corresponding files. The `primaryDataset` key for Data datasets is needed in order to apply a trigger
selection only to the corresponding dataset (e.g. apply the `SingleMuon` trigger only to datasets having
`primaryDataset=SingleMuon`).

The structure of the ``datasets_definitions.json`` file after filling in the dictionary with the parameters relevant to
our Drell-Yan and SingleMuon datasets should be the following:

```json 
{
   "DYJetsToLL_M-50":{
        "sample": "DYJetsToLL",
        "json_output"    : "datasets/DYJetsToLL_M-50.json",
        "files":[
            { "das_names": ["/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM"],
              "metadata": {
                  "year":"2018",
                  "isMC": true,
		          "xsec": 6077.22,
                  "part": "" 
                  }
            }
        ]
  },
    "DATA_SingleMuon": {
        "sample": "DATA_SingleMuonC",
        "json_output": "datasets/DATA_SingleMuonC.json",
        "files": [

            {
                "das_names": [
                    "/SingleMuon/Run2018C-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD"
                ],
                "metadata": {
                    "year": "2018",
                    "isMC": false,
                    "primaryDataset": "SingleMuon",
                    "era": "C"
                },
                "das_parents_names": [
                    "/SingleMuon/Run2018C-UL2018_MiniAODv2-v2/MINIAOD"
                ]
            }
        ]
    }
}
```

2) To produce the json files containing the file lists, run the following command:

```bash
voms-proxy-init -voms cms -rfc --valid 168:0

build_dataset.py --cfg datasets/dataset_definitions.json
```

Four ``json`` files are produced as output, two for each dataset: a version includes file paths with a specific prefix
corresponding to a site (querying Rucio for the availability of the files at the moment of running the script)
while another has a global redirector prefix (e.g. ``xrootd-cms.infn.it``), and is named with the suffix
`_redirector.json` If one has to rebuild the dataset to include more datasets, the extra argument ``--overwrite`` can be
provided to the script.

```bash
ls zmumu/datasets
  datasets_definitions.json DATA_SingleMuonC.json DATA_SingleMuonC_redirector.json DYJetsToLL_M-50.json
  DYJetsToLL_M-50_redirector.json
```


# Analysis configuration
## Define selections

The selections are performed at two levels:

* Object preselection: selecting the "good" objects that will be used in the final analysis (e.g. `JetGood`, `MuonGood`, `ElectronGood`...).
* Event selection: selections on the events that enter the final analysis, done in three steps:

   1) Skim and trigger: loose cut on the events and trigger requirements.
   2) Preselection: baseline event selection for the analysis.
   3) Categorization: selection to split the events passing the event preselection into different categories (e.g. signal region, control region).

## Object preselection

To select the objects entering the final analysis, we need to specify a series of cut parameters for the leptons and
jets in the file ``params/object_preselection.yaml``. These selections include the pT, eta
acceptance cuts, the object identification working points, the muon isolation, the b-tagging working point, etc.

For the Z->mumu analysis, we just use the standard definitions for the muon, electron and jet objects:

```yaml
object_preselection:
  Muon:
    pt: 15
    eta: 2.4
    iso: 0.25  #PFIsoLoose
    id: tightId

  Electron:
    pt: 15
    eta: 2.4
    iso: 0.06
    id: mvaFall17V2Iso_WP80

  Jet:
    dr_lepton: 0.4
    pt: 30
    eta: 2.4
    jetId: 2
    puId:
      wp: L
      value: 4
      maxpt: 50.0

```

## Event selection

In PocketCoffea, the event selections are implemented with a dedicated `Cut` object, that stores both the information of the cutting function and its input parameters.
Several factory ``Cut`` objects are available in ``pocket_coffea.lib.cut_functions``, otherwise the user can define their own custom ``Cut`` objects.


### Skim

The skim selection of the events is performed "on the fly" to reduce the number of processed events. At this stage we
also apply the HLT trigger requirements required by the analysis.  The following steps of the analysis are performed
only on the events passing the skim selection, while the others are discarded from the branch ``events``, therefore
reducing the computational load on the processor.  In the config file, we specify two skim cuts: one is selecting events
with at least one 15 GeV muon and the second is requiring the HLT ``SingleMuon`` path.

In the preamble of ``config.py``, we define our custom trigger dictionary, which we pass as an argument to the factory function ``get_HLTsel()``:

```python
   trigger_dict = {
      "2018": {
         "SingleEle": [
               "Ele32_WPTight_Gsf",
               "Ele28_eta2p1_WPTight_Gsf_HT150",
         ],
         "SingleMuon": [
               "IsoMu24",
         ],
      },
   }

   cfg = {
    ...
    "skim": [get_nObj_min(1, 15., "Muon"),
             get_HLTsel(primaryDatasets=["SingleMuon"])],
    ...
```


### Event preselection

In the Z->mumu analysis, we want to select events with exactly two muons with opposite charge. In addition, we require a
cut on the leading muon pT and on the dilepton invariant mass, to select the Z boson mass window.  The parameters are
directly passed to the constructor of the ``Cut`` object as the dictionary ``params``. We can define the function
``dimuon`` and the ``Cut`` object ``dimuon_presel`` in the preamble of the config script:

```python
   def dimuon(events, params, year, sample, **kwargs):

      # Masks for same-flavor (SF) and opposite-sign (OS)
      SF = ((events.nMuonGood == 2) & (events.nElectronGood == 0))
      OS = events.ll.charge == 0

      mask = (
         (events.nLeptonGood == 2)
         & (ak.firsts(events.MuonGood.pt) > params["pt_leading_muon"])
         & OS & SF
         & (events.ll.mass > params["mll"]["low"])
         & (events.ll.mass < params["mll"]["high"])
      )

      # Pad None values with False
      return ak.where(ak.is_none(mask), False, mask)

   dimuon_presel = Cut(
      name="dilepton",
      params={
         "pt_leading_muon": 25,
         "mll": {'low': 25, 'high': 2000},
      },
      function=dimuon,
   )
```

In a scenario of an analysis requiring several different cuts, a dedicated library of cuts and functions can be defined in a separate file and imported in the config file.

The ``preselections`` field in the config file is updated accordingly:

```python

   cfg = {
      ...
      "preselections" : [dimuon_presel],
      ...
   }
```

### Categorization

In the toy Z->mumu analysis, no further categorization of the events is performed. Only a ``baseline`` category is
defined with the ``passthrough`` factory cut that is just passing the events through without any further selection:

```python
   cfg = {
      ...
      # Cuts and plots settings
      "finalstate" : "dimuon",
      "skim": [get_nObj_min(1, 15., "Muon"),
               get_HLTsel("dimuon", trigger_dict, primaryDatasets=["SingleMuon"])],
      "preselections" : [dimuon_presel],
      "categories": {
         "baseline": [passthrough],
      },
      ...
   }
   ```

If for example Z->ee events were also included in the analysis, one could have defined a more general "dilepton"
preselection and categorized the events as ``2e`` or ``2mu`` depending if they contain two electrons or two muons,
respectively.

## Define weights and variations

The application of the nominal value of scale factors and weights is switched on and off just by adding the corresponding key in the ``weights`` dictionary:

```

   cfg = {
      ...
      "weights": {
         "common": {
            "inclusive": ["genWeight","lumi","XS",
                          "pileup",
                          "sf_mu_id","sf_mu_iso",
                          ],
            "bycategory" : {
            }
        },
        "bysample": {
        }
      },
      ...
   }
```

In our case, we are applying the nominal scaling of Monte Carlo by ``lumi * XS / genWeight`` together with the pileup reweighting and the muon ID and isolation scale factors.
The reweighting of the events is managed internally by the module ``WeightsManager``.

To store also the up and down systematic variations corresponding to a given weight, one can specify it in the ``variations`` dictionary:

```python

   cfg = {
      ...
      "variations": {
         "weights": {
            "common": {
               "inclusive": [ "pileup",
                              "sf_mu_id", "sf_mu_iso"
                           ],
               "bycategory" : {
               }
            },
         "bysample": {
         }  
         },  
      },
      ...
   }
```

In this case we will store the variations corresponding to the systematic variation of pileup and the muon ID and isolation scale factors.
These systematic uncertainties will be included in the final plots.

## Define histograms

In PocketCoffea histograms can be defined directly from the configuration:  look at the ``variable`` dictionary under
``config.py``.
Histograms are defined with the options specified in
[`hist_manager.py`](https://github.com/PocketCoffea/PocketCoffea/blob/main/pocket_coffea/lib/hist_manager.py#L30). An
histogram is a collection of `Axis` objects with additional options for excluding/including variables, samples, and
systematic variations. 

In order to create a user defined histogram add `Axis` as a list (1 element for 1D-hist, 2 elements for 2D-hist)

```python
   "variables":
       {
           # 1D plots
           "ll" : HistConf( [Axis(coll="ll", field="mass", bins=100, start=50, stop=150, label=r"$M_{\ell\ell}$ [GeV]")] 
    	}
	
	# coll : collection/objects under events
	# field: fields under collections
	# bins, start, stop: # bins, axis-min, axis-max
	# label: axis label name
```

The `collection` is the name used to access the fields of the `events` main dataset (the NanoAOD Events tree). The `field` specifies the specific
array to use. If a field is global in the `events`, e.g. a user defined arrays in events, just use `coll=events`. Please
refer to [the `Axis` code](https://github.com/PocketCoffea/PocketCoffea/blob/main/pocket_coffea/lib/hist_manager.py#L12) for a complete description of the options. 


- There are some predefined `hist`_. 

```python

	"variables":
       {
        **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
	# Muon kinematics
	**muon_hists(coll="MuonGood", pos=0),
	# Jet kinematics
        **jet_hists(coll="JetGood", pos=0),
    	}
```

	
# Run the processor

Run the coffea processor to get ``.coffea`` output files. The ``coffea`` processor can be run locally with ``iterative`` or ``futures`` executors or scaleout to clusters. We now test the setup on ``lxplus``, ``naf-desy`` but more sites can also be included later.

```bash
# read all information from the config file
runner.py --cfg example_config.py --full  -o output_v1

# iterative run is also possible
## run --test for iterative processor with ``--limit-chunks/-lc``(default:2) and ``--limit-files/-lf``(default:1)
runner.py --cfg example_config.py  --full --test --lf 1 --lc  2 -o output_v1

## change the --executor and numbers of jobs with -s/--scaleout
runner.py --cfg example_config.py  --full --executor futures -s 10 -o output_v1
```

The scaleout configurations really depends on cluster and schedulers with different sites(lxplus, LPC, naf-desy).

```python

## Example for naf-desy
run_options = {
    "executor"       : "parsl/condor/naf-desy", # scheduler/cluster-type/site
    "workers"        : 1, # cpus for each job
    "scaleout"       : 300, # numbers of job
    "queue"          : "microcentury",# job queue time for condor
    "walltime"       : "00:40:00", # walltime for condor jobs
    "disk_per_worker": "4GB", # disk size for each job(stored files)
    "mem_per_worker" : "2GB", # RAM size for each job
    "exclusive"      : False, # not used for condor
    "chunk"          : 200000, #chunk size 
    "retries"        : 20, # numbers of retries when job failes
    "max"            : None, # numbers of chunks 
    "skipbadfiles"   : None, # skip badfiles
    "voms"           : None, # point to the voms certificate directory
    "limit"          : None, # limited files
    }
```

The output of the script will be similar to 

```bash
$ runner.py --cfg example_config.py -o output_all

    ____             __        __  ______      ________          
   / __ \____  _____/ /_____  / /_/ ____/___  / __/ __/__  ____ _
  / /_/ / __ \/ ___/ //_/ _ \/ __/ /   / __ \/ /_/ /_/ _ \/ __ `/
 / ____/ /_/ / /__/ ,< /  __/ /_/ /___/ /_/ / __/ __/  __/ /_/ / 
/_/    \____/\___/_/|_|\___/\__/\____/\____/_/ /_/  \___/\__,_/  
                                                                 

Loading the configuration file...
[INFO    ] Configurator instance:
  - Workflow: <class 'workflow.ZmumuBaseProcessor'>
  - N. datasets: 5 
   -- Dataset: DATA_SingleMuon_2018_EraA,  Sample: DATA_SingleMuon, N. files: 92, N. events: 241608232
   -- Dataset: DATA_SingleMuon_2018_EraB,  Sample: DATA_SingleMuon, N. files: 51, N. events: 119918017
   -- Dataset: DATA_SingleMuon_2018_EraC,  Sample: DATA_SingleMuon, N. files: 56, N. events: 109986009
   -- Dataset: DATA_SingleMuon_2018_EraD,  Sample: DATA_SingleMuon, N. files: 194, N. events: 513909894
   -- Dataset: DYJetsToLL_M-50_2018,  Sample: DYJetsToLL, N. files: 204, N. events: 195510810
  - Subsamples:
   -- Sample DATA_SingleMuon: StandardSelection ['DATA_SingleMuon'], (1 categories)
   -- Sample DYJetsToLL: StandardSelection ['DYJetsToLL'], (1 categories)
  - Skim: ['nMuon_min1_pt18.0', 'HLT_trigger_SingleMuon']
  - Preselection: ['dilepton']
  - Categories: StandardSelection ['baseline'], (1 categories)
  - Variables:  ['MuonGood_eta_1', 'MuonGood_pt_1', 'MuonGood_phi_1', 'nElectronGood', 'nMuonGood', 'nJets', 'nBJets', 'JetGood_eta_1', 'JetGood_pt_1', 'JetGood_phi_1', 'JetGood_btagDeepFlavB_1', 'JetGood_eta_2', 'JetGood_pt_2', 'JetGood_phi_2', 'JetGood_btagDeepFlavB_2', 'mll']
  - available weights variations: {'DATA_SingleMuon': ['nominal', 'sf_mu_iso', 'pileup', 'sf_mu_id'], 'DYJetsToLL': ['nominal', 'sf_mu_iso', 'pileup', 'sf_mu_id']} 
  - available shape variations: {'DATA_SingleMuon': [], 'DYJetsToLL': []}
Saving config file to output_all/config.json
....
```


## Produce plots

To be completed
