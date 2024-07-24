import awkward as ak

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.lib.objects import lepton_selection
from pocket_coffea.utils.configurator import Configurator


class ElectronProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)


    def apply_object_preselection(self, variation):
        '''
        Cleaning only Electrons
        '''
        # Include the supercluster pseudorapidity variable
        electron_etaSC = self.events.Electron.eta + self.events.Electron.deltaEtaSC
        self.events["Electron"] = ak.with_field(
            self.events.Electron, electron_etaSC, "etaSC"
        )
        # We don't use the lepton_selection utils, because we just cut by pt
        self.events["ElectronGood"] = self.events.Electron[self.events.Electron.pt > self.params.object_preselection.Electron.pt]
        

    def count_objects(self, variation):
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood)

