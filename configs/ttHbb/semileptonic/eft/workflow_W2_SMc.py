#Workflow to study the list 2 of wilson coefficients

import numpy as np
import awkward as ak

from coffea.analysis_tools import PackedSelection
from pocket_coffea.lib.deltaR_matching import metric_eta, metric_phi, metric_pt

from pocket_coffea.workflows.base import BaseProcessorABC
#import eft_weights

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




    def particle_selection(self):

        is_higgs_mask = self.events.GenPart.pdgId == 25
        is_top_mask = self.events.GenPart.pdgId == 6
        is_antitop_mask = self.events.GenPart.pdgId == -6

        #Optimized mask to identifiy the correct particles from GenPart
        #In GenPart the last copy of the particle is the correct one (the one with children)

        mask_flags = self.events.GenPart.hasFlags(['isPrompt', 'fromHardProcess', 'isHardProcess'])

        #mask_children = ak.num(self.events.GenPart[is_higgs_mask].childrenIdxG, axis=2) == 2 

        higgs = ak.flatten(self.events.GenPart[is_higgs_mask & mask_flags])
        top = self.events.GenPart[is_top_mask & mask_flags]
        antitop = self.events.GenPart[is_antitop_mask & mask_flags]


        #self.events.HiggsParton cointains Higgs particles at GenPart level
        #self.events.TopParton cointains top particles at GenPart level
        #self.events.AntiTopParton cointains antitop particles at GenPart level
        #self.events.nGenJet cointains the number of GenJets


        self.events["HiggsParton"] = higgs 
        self.events["TopParton"] = top 
        self.events["AntiTopParton"] = antitop 


        #deltaR between higgs and top

        deltaR_ht = higgs.delta_r(top)

        #deltaR between higgs and antitop

        deltaR_hat = higgs.delta_r(antitop)

        #self.events["deltaR_ht_min"] = ak.min(deltaR_ht, axis=1)
        #self.events["deltaR_hat_min"] = ak.min(deltaR_hat, axis=1)

        self.events["deltaR_ht"] = deltaR_ht
        self.events["deltaR_hat"] = deltaR_ht



        #Number of GenJet

        nGenJet = ak.num(self.events.GenJet.mass)  #I just need to count the number of masses (or also other variables) in GenJet

        self.events["nGenJet"] = nGenJet

        
        #delta_phi between the two top

        deltaPhi = top.delta_phi(antitop)
        deltaPhi_abs = abs(top.phi-antitop.phi)

        self.events["deltaPhi_tt"] = deltaPhi
        self.events["deltaPhi_tt_abs"] = deltaPhi_abs


        #delta_eta between the two top

        deltaEta = abs(top.eta-antitop.eta)

        self.events["deltaEta_tt"] = deltaEta


        #delta_pt and sum_pt between the two top

        deltaPt = abs(top.pt-antitop.pt)

        self.events["deltaPt_tt"] = deltaPt

        sumPt = (top+antitop).pt 
 
        self.events["sumPt_tt"] = sumPt


        #Weights 

        self.events["cthre_BSMc"] = self.events.LHEReweightingWeight[:,5]
        self.events["ctwre_BSMc"] = self.events.LHEReweightingWeight[:,11]
        self.events["ctbre_BSMc"] = self.events.LHEReweightingWeight[:,17]
        self.events["cbwre_BSMc"] = self.events.LHEReweightingWeight[:,23]
        self.events["chq1_BSMc"] = self.events.LHEReweightingWeight[:,29]
        self.events["chq3_BSMc"] = self.events.LHEReweightingWeight[:,35]
        self.events["cht_BSMc"] = self.events.LHEReweightingWeight[:,41]
        self.events["chtbre_BSMc"] = self.events.LHEReweightingWeight[:,47]

        self.events["cthre_SMc"] = self.events.LHEReweightingWeight[:,4]
        self.events["ctwre_SMc"] = self.events.LHEReweightingWeight[:,10]
        self.events["ctbre_SMc"] = self.events.LHEReweightingWeight[:,16]
        self.events["cbwre_SMc"] = self.events.LHEReweightingWeight[:,22]
        self.events["chq1_SMc"] = self.events.LHEReweightingWeight[:,28]
        self.events["chq3_SMc"] = self.events.LHEReweightingWeight[:,34]
        self.events["cht_SMc"] = self.events.LHEReweightingWeight[:,40]
        self.events["chtbre_SMc"] = self.events.LHEReweightingWeight[:,46]


       

        


    def process_extra_after_presel(self, variation) -> ak.Array:
        # Compute the structure of the EFT weights for all the events

        self.particle_selection()
