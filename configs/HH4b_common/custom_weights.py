from pocket_coffea.lib.weights.weights import WeightLambda

bkg_morphing_dnn_weight = WeightLambda.wrap_func(
    name="bkg_morphing_dnn_weight",
    function=lambda params, metadata, events, size, shape_variations: events.bkg_morphing_dnn_weight,
    has_variations=False,
    isMC_only=False,
)
bkg_morphing_dnn_weightRun2 = WeightLambda.wrap_func(
    name="bkg_morphing_dnn_weightRun2",
    function=lambda params, metadata, events, size, shape_variations: events.bkg_morphing_dnn_weightRun2,
    has_variations=False,
    isMC_only=False,
)
