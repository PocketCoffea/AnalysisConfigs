import awkward as ak

from configs.ttHbb.semileptonic.common.workflows.workflow_tthbb import ttHbbPartonMatchingProcessor
from pocket_coffea.objects.jets import met_xy_correction

class METxyProcessor(ttHbbPartonMatchingProcessor):
    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation)
        met_pt_corr, met_phi_corr = met_xy_correction(self.params, self.events, self._year, self._era)
        self.events["MET_corr"] = ak.zip(
            {
                "pt": met_pt_corr,
                "phi": met_phi_corr,
            }
        )