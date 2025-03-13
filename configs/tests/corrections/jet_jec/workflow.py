import sys
import awkward as ak

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.lib.objects import lepton_selection
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.objects import jet_correction_correctionlib
import pocket_coffea.lib.jets as jets_module

# Ref: https://gitlab.cern.ch/cms-analysis/general/analysis_recipes/-/blob/master/frameworks/cmsjmecalculators/run_recipe.sh?ref_type=heads
JECversions = {
    '2016_PreVFP':   'Summer19UL16APV_V7_MC', 
    '2016_PostVFP':  'Summer19UL16_V7_MC', 
    '2017':          'Summer19UL17_V5_MC', 
    '2018':          'Summer19UL18_V5_MC', 
    '2022_preEE':    'Summer22_22Sep2023_V2_MC',
    '2022_postEE':   'Summer22EE_22Sep2023_V2_MC',
    '2023_preBPix':  'Summer23Prompt23_V1_MC',
    '2023_postBPix': 'Summer23BPixPrompt23_V1_MC'
}


# Ref: https://github.com/PocketCoffea/PocketCoffea/blob/main/pocket_coffea/parameters/jec_config.py
JECjsonFiles = {
    '2016_PreVFP':   {'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016preVFP_UL/jet_jerc.json.gz'},
    '2016_PostVFP':  {'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016postVFP_UL/jet_jerc.json.gz'},
    '2017':          {'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2017_UL/jet_jerc.json.gz'},
    '2018':          {'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/jet_jerc.json.gz'},
    '2022_preEE':    {'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2022_Summer22/jet_jerc.json.gz'},
    '2022_postEE':   {'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2022_Summer22EE/jet_jerc.json.gz'},
    '2023_preBPix':  {'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2023_Summer23/jet_jerc.json.gz'},
    '2023_postBPix': {'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2023_Summer23BPix/jet_jerc.json.gz'}
}

jets_module.JECjsonFiles = JECjsonFiles


class JETJEC(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)


    def apply_object_preselection(self, variation):
        '''
        Extracts corrected jets.
        '''
        
        _era = self.events.metadata["year"]
        _year = int(_era[:4])
        typeJet = "AK4PFchs" if _year < 2019 else "AK4PFPuppi"
        
        jet_out = jet_correction_correctionlib(self.events, "Jet",
                                            typeJet, self.events.metadata["year"],
                                            JECversions[_era], JERversion=None, verbose=False)
        
        self.events["JetsCorrected"] = jet_out        

    def count_objects(self, variation):
        self.events["nJetsCorrected"] = ak.num(self.events.JetsCorrected)

