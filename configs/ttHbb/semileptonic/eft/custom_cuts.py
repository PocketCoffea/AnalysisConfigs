from pocket_coffea.lib.cut_definition import Cut  
import awkward as ak


def cut_weight(events, params, year, sample, **kwargs):
    pass 
    return events.LHEReweightingWeight[:,params["selected_weight"]]>params["max_weight"]


def cut_ev(events, params, year, sample, **kwargs):

    mask_status = events.LHEPart.status==1
    parts_out = events.LHEPart[mask_status]
    mask_q = abs(parts_out.pdgId) < 11 
    mask_g = parts_out.pdgId == 21

    mask_e = abs(parts_out.pdgId) == 11
    mask_mu = abs(parts_out.pdgId) == 13
    mask_tau = abs(parts_out.pdgId) == 15

    parts_j = parts_out[mask_q | mask_g]
    parts_l = parts_out[mask_e | mask_mu | mask_tau]


    mask_j_pt = parts_j.pt < params["j_pt_min"]
    mask_j_eta = abs(parts_j.eta) > params["j_eta_max"]

    mask_l_pt = parts_l.pt < params["l_pt_min"]
    mask_l_eta = abs(parts_l.eta) > params["l_eta_max"]


    mask_ev_jets = ak.sum((mask_j_pt | mask_j_eta), axis=1) == 0
    mask_ev_lep = ak.sum((mask_l_pt | mask_l_eta), axis=1) == 0

    return mask_ev_jets | mask_ev_lep

cut_ctwre = Cut(
    name = "",
    params = {
        "selected_weight":11,
        "max_weight":3,
        },
    function=cut_weight
)

cut_cbwre = Cut(
    name = "",
    params = {
        "selected_weight":13,
        "max_weight":3,
        },
    function=cut_weight
)


cut_ctbre = Cut(
    name = "",
    params = {
        "selected_weight":17,
        "max_weight":3,
        },
    function=cut_weight
)


cut_events = Cut(
     name= "",
     params = {
        "j_pt_min":15.,
        "j_eta_max":4.5,
        "l_pt_min":15.,
        "l_eta_max":3,
     },

     function=cut_ev


)