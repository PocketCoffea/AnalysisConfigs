from pocket_coffea.lib.cut_definition import Cut  

def cut_function(events, params, year, sample, **kwargs):
    pass 
    return events.LHEReweightingWeight[:,params["selected_weight"]]>params["max_weight"]

cut_ctwre = Cut(
    name = "",
    params = {
        "selected_weight":11,
        "max_weight":3,
        },
    function=cut_function
)

cut_cbwre = Cut(
    name = "",
    params = {
        "selected_weight":13,
        "max_weight":3,
        },
    function=cut_function
)


cut_ctbre = Cut(
    name = "",
    params = {
        "selected_weight":17,
        "max_weight":3,
        },
    function=cut_function
)