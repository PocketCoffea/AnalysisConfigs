import sys
import awkward as ak

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.objects import jet_correction_correctionlib
import pocket_coffea.lib.jets as jets_module


class DoNothing(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    def apply_object_preselection(self, variation):
        pass

    def count_objects(self, variation):
        pass

