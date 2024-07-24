import awkward as ak

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.lib.objects import lepton_selection
from pocket_coffea.utils.configurator import Configurator


class MuonProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)


    def apply_object_preselection(self, variation):
        '''
        Cleaning only Electrons
        '''
        muon_mask = ((self.events.Muon.pt >= self.params.object_preselection.Muon.pt) & 
                    (abs(self.events.Muon.eta) <= self.params.object_preselection.Muon.eta))
        self.events["MuonGood"] = self.events.Muon[muon_mask]
        

    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)

