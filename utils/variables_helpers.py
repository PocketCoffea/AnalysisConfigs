from pocket_coffea.parameters.histograms import jet_hists
from pocket_coffea.lib.hist_manager import HistConf, Axis

# Combine jet_hists from position start to position end
def jet_hists_dict(coll="JetGood", start=1, end=5):
    combined_dict = {}
    for pos in range(start, end + 1):
        combined_dict.update(
            jet_hists(coll=coll, pos=pos)
        )
    return combined_dict


# Helper function to create HistConf() for a specific configuration
def create_HistConf(coll, field, pos=None, bins=60, start=0, stop=1, label=None):
    axis_params = {
        "coll": coll,
        "field": field,
        "bins": bins,
        "start": start,
        "stop": stop,
        "label": label if label else field,
    }
    if pos is not None:
        axis_params["pos"] = pos
    return {label: HistConf([Axis(**axis_params)])}
