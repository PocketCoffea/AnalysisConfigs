from pocket_coffea.workflows.base import BaseProcessorABC
import awkward as ak

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

        

    def count_objects(self, variation):
        pass

    
