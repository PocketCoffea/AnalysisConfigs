import awkward as ak
from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.lib.deltaR_matching import metric_eta, metric_phi
from pocket_coffea.lib.deltaR_matching import object_matching
from pocket_coffea.lib.jets import jet_selection

from custom_cut_functions import *
from custom_cuts import *

class hh4bProcessor(BaseProcessorABC):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)

    def apply_object_preselection(self, variation):
        # super().apply_object_preselection(variation=variation)

        self.events["JetGood"]= jet_selection_nopu(self.events, "Jet", self.params)
        self.events["ElectronGood"] = lepton_selection(self.events, "Electron", self.params)
        self.events["MuonGood"] = lepton_selection(self.events, "Muon", self.params)

        self.events["JetGoodBtagOrdered"] = self.events.JetGood[
            ak.argsort(self.events.JetGood.btagDeepFlavB, axis=1, ascending=False)
        ] # TODO: use particlenet!




    def count_objects(self, variation):
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood, axis=1)
        self.events["nMuonGood"] = ak.num(self.events.MuonGood, axis=1)
        self.events["nJetGood"] = ak.num(self.events.JetGood, axis=1)

    # def process_extra_after_presel(self, variation) -> ak.Array:
    #     self.count_leptons()
