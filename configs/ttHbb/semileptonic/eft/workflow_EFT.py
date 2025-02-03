from configs.ttHbb.semileptonic.common.workflows.workflow_ttHbb_genmatching_full import ttHbbPartonMatchingProcessorFull
from eft_weights import EFTStructure
import awkward as ak

class ttHbbEFTProcessor(ttHbbPartonMatchingProcessorFull):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        # Initialize info for EFT
        self.eft_structure = EFTStructure(self.params.eft.reweight_card)
                

    def process_extra_after_presel(self, variation):
        super().process_extra_after_presel(variation=variation)
        if self._sample == "ttHTobb_EFT":
            self.events = ak.with_field(self.events,self.eft_structure.get_structure_constants(ak.to_numpy(self.events["LHEReweightingWeight"])), "EFT_struct")
            
            print("EFT struct: ", self.events["EFT_struct"])
            print("structure constants: ", self.eft_structure.get_structure_constants(ak.to_numpy(self.events["LHEReweightingWeight"])))
            breakpoint()

