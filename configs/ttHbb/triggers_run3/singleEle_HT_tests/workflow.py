from pocket_coffea.workflows.base import BaseProcessorABC
import awkward as ak
from pocket_coffea.lib.objects import lepton_selection


class TriggerProcessor(BaseProcessorABC):

    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)

    def apply_object_preselection(self, variation):
        leptons = ak.with_name(
            ak.concatenate((self.events.Muon, self.events.Electron), axis=1),
            name='PtEtaPhiMCandidate',
        )
        self.events["Lepton"] = leptons[ak.argsort(leptons.pt, ascending=False)]

        higgs = self.events.LHEPart[(self.events.LHEPart.status==1)&(self.events.LHEPart.pdgId==25)]
        self.events["higgs"] = higgs

        # Include the supercluster pseudorapidity variable
        electron_etaSC = self.events.Electron.eta + self.events.Electron.deltaEtaSC
        self.events["Electron"] = ak.with_field(
            self.events.Electron, electron_etaSC, "etaSC"
        )
        # Build masks for selection of muons, electrons, jets, fatjets
        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
        )

        self.events["recoHT"] = ak.sum(self.events.Jet.pt, axis=1) + ak.sum(self.events.ElectronGood.pt, axis=1)
        

    def count_objects(self, variation):
        self.events["nJets"] = ak.num(self.events.Jet)

    
