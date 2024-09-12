import awkward as ak
import workflow_spanet
from workflow_spanet import SpanetInferenceProcessor
import quantile_transformer
from quantile_transformer import WeightedQuantileTransformer

class ControlRegionsProcessor(SpanetInferenceProcessor):

    def process_extra_after_presel(self, variation) -> ak.Array:
        super().process_extra_after_presel(variation)

        params_quantile_transformer = self.params["quantile_transformer"][self.events.metadata["year"]]
        transformer = WeightedQuantileTransformer(n_quantiles=params_quantile_transformer["n_quantiles"], output_distribution=params_quantile_transformer["output_distribution"])
        transformer.load(params_quantile_transformer["file"])
        transformed_score = transformer.transform(self.events.spanet_output.tthbb)
        self.events["spanet_output"] = ak.with_field(self.events["spanet_output"], transformed_score, "tthbb_transformed")
