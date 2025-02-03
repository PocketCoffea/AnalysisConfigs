import awkward as ak
import sys
import numpy as np

sys.path.append("../")
from HH4b_common.workflow_common import HH4bCommonProcessor


class HH4bbQuarkMatchingProcessor(HH4bCommonProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.random_pt = self.workflow_options["random_pt"]
        self.rand_type = self.workflow_options["rand_type"]

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation)

    def process_extra_after_presel(self, variation):  # -> ak.Array
        self.flatten_pt(variation)
        super().process_extra_after_presel(variation)

    def flatten_pt(self, variation):
        if self._isMC and not self.classification:
            if self.random_pt:  # TODO implement random_pt
                if self.rand_type == 0.5:
                    random_weights = ak.Array(np.random.rand((len(self.events["nJetGood"])))+0.5)  # [0.5,1.5]
                elif self.rand_type == 0.3:
                    random_weights = ak.Array(np.random.rand((len(self.events["nJetGood"])))*1.4+0.3)  # [0.3,1.7]
                elif self.rand_type == 0.1:
                    random_weights = ak.Array(np.random.rand((len(self.events["nJetGood"])))*9.9+0.1)  # [0.3,1.7]
                else:
                    raise ValueError(f"Invalid input. self.rand_type {self.rand_type} not known.")
                self.events = ak.with_field(
                        self.events,
                        random_weights,
                        "random_pt_weights",
                        )
                self.events["JetGoodHiggs"] = ak.with_field(
                        self.events.JetGoodHiggs,
                        self.events.JetGoodHiggs.mass,
                        "mass_orig",
                        )
                self.events["JetGoodHiggs"] = ak.with_field(
                        self.events.JetGoodHiggs,
                        self.events.JetGoodHiggs.mass*random_weights,
                        "mass",
                        )
                self.events["JetGoodHiggs"] = ak.with_field(
                        self.events.JetGoodHiggs,
                        self.events.JetGoodHiggs.pt,
                        "pt_orig",
                        )
                self.events["JetGoodHiggs"] = ak.with_field(
                        self.events.JetGoodHiggs,
                        self.events.JetGoodHiggs.pt*random_weights,
                        "pt",
                        )
                self.events["JetGood"] = ak.with_field(
                        self.events.JetGood,
                        self.events.JetGood.mass,
                        "mass_orig",
                        )
                self.events["JetGood"] = ak.with_field(
                        self.events.JetGood,
                        self.events.JetGood.mass*random_weights,
                        "mass",
                        )
                self.events["JetGood"] = ak.with_field(
                        self.events.JetGood,
                        self.events.JetGood.pt,
                        "pt_orig",
                        )
                self.events["JetGood"] = ak.with_field(
                        self.events.JetGood,
                        self.events.JetGood.pt*random_weights,
                        "pt",
                        )
    
