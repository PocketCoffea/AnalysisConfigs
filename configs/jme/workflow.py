import awkward as ak

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.hist_manager import Axis
from pocket_coffea.lib.objects import (
    jet_correction,
    lepton_selection,
    jet_selection,
    btagging,
    get_dilepton,
)
from custom_cut_functions import jet_selection_nopu


class QCDBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)


    def apply_object_preselection(self, variation):
        self.events["JetGood"], self.jetGoodMask = jet_selection_nopu(
            self.events, "Jet", self.params#, "LeptonGood"
        )
        # self.events["GenJetGood"], self.genjetGoodMask = jet_selection_nopu(
        #     self.events, "GenJet", self.params#, "LeptonGood"
        # )

        # self.events["JetGood"], self.jetGoodMask = self.events["Jet"], ak.ones_like(self.events["Jet"].pt, dtype=bool)
        # if True: #self._isMC:
        #     self.events["GenJetGood"] = self.events.GenJet
        #     # find jetgoodmatched to genjetgood
        #     self.events["GenJetGoodMatched"] = self.events.GenJetGood[self.events.JetGood.genJetIdx>=0]


        # if True: #self._isMC:
        #     Ngenjet = ak.num(self.events.GenJet)
        #     matching_with_none = ak.mask(self.events.Jet.genJetIdx, (self.events.Jet.genJetIdx < Ngenjet)&(self.events.Jet.genJetIdx!=-1))
        #     self.events["GenJetsOrder"] = self.events.GenJet[matching_with_none]
        #     not_none = ~ak.is_none(self.events.GenJetsOrder, axis=1)
        #     self.events["JetMatched"] = ak.mask(self.events.Jet, not_none)


    def count_objects(self, variation):
        pass
        # self.events["nJetGood"] = ak.num(self.events.JetGood)
        # if True: #self._isMC:
        #     self.events["nGenJetGood"] = ak.num(self.events.GenJetGood)

    # Function that defines common variables employed in analyses and save them as attributes of `events`
    def define_common_variables_after_presel(self, variation):
        # if True: #self._isMC:
        #     self.events["JetMatched"] = ak.with_field(self.events.JetMatched, self.events.JetMatched.pt/self.events.GenJetsOrder.pt, "Response")
        pass
        # set the puId for jets to 1
        # self.events["Jet_puId"] = ak.ones_like(self.events.JetGood.pt)

        # self.events["JetGood_Ht"] = ak.sum(abs(self.events.JetGood.pt), axis=1)
