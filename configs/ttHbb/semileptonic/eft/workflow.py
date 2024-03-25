import numpy as np
import awkward as ak

from coffea.analysis_tools import PackedSelection

from pocket_coffea.workflows.base import BaseProcessorABC

class BaseProcessorGen(BaseProcessorABC):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)

    def skim_events(self):
        '''
        We redefine the skim_events method such that the event flags and PV requirements are not applied.
        '''
        self._skim_masks = PackedSelection()

        for skim_func in self._skim:
            # Apply the skim function and add it to the mask
            mask = skim_func.get_mask(
                self.events,
                processor_params=self.params,
                year=self._year,
                sample=self._sample,
                isMC=self._isMC,
            )
            self._skim_masks.add(skim_func.id, mask)

        # Finally we skim the events and count them
        self.events = self.events[self._skim_masks.all(*self._skim_masks.names)]
        self.nEvents_after_skim = self.nevents
        self.output['cutflow']['skim'][self._dataset] = self.nEvents_after_skim
        self.has_events = self.nEvents_after_skim > 0

    def apply_object_preselection(self, variation):
        '''
        We redefine the abstract method apply_object_preselection.
        This method is called by the BaseProcessorABC to apply the object preselection
        and it can be modified to define the custom object preselection of the analysis.
        '''
        pass

    def count_objects(self, variation):
        '''
        We redefine the abstract method count_objects.
        This method is called by the BaseProcessorABC to count the objects of the analysis
        defined in the object preselection.
        '''
        pass