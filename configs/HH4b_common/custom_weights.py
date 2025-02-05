from pocket_coffea.lib.weights.weights import WeightLambda

bkg_morhping_dnn_weight= WeightLambda.wrap_func(
    name="bkg_morhping_dnn_weight",
    function=lambda params, metadata, events, size, shape_variations:
            events.events.bkg_morhping_dnn_weight,
    has_variations=False
    )