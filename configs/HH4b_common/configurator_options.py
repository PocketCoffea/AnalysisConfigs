import sys
sys.path.append("../../")
from utils.variables_helpers import jet_hists_dict, create_HistConf

from pocket_coffea.lib.columns_manager import ColOut

variables_dict = {
        # **count_hist(coll="JetGood", bins=10, start=0, stop=10),
        # **count_hist(coll="JetGoodHiggs", bins=10, start=0, stop=10),
        # **count_hist(coll="ElectronGood", bins=3, start=0, stop=3),
        # **count_hist(coll="MuonGood", bins=3, start=0, stop=3),
        # **count_hist(coll="JetGoodHiggsMatched", bins=10, start=0, stop=10),
        # **count_hist(coll="bQuarkHiggsMatched", bins=10, start=0, stop=10),
        # **count_hist(coll="JetGoodMatched", bins=10, start=0, stop=10),
        # **count_hist(coll="bQuarkMatched", bins=10, start=0, stop=10),
        # **jet_hists(coll="JetGood", pos=0),
        # **jet_hists(coll="JetGood", pos=1),
        # **jet_hists(coll="JetGood", pos=2),
        # **jet_hists(coll="JetGood", pos=3),
        # **jet_hists(coll="JetGoodHiggsPtOrder", pos=0),
        # **jet_hists(coll="JetGoodHiggsPtOrder", pos=1),
        # **jet_hists(coll="JetGoodHiggsPtOrder", pos=2),
        # **jet_hists(coll="JetGoodHiggsPtOrder", pos=3),
        **jet_hists(coll="JetGoodHiggs", pos=0),
        **jet_hists(coll="JetGoodHiggs", pos=1),
        **jet_hists(coll="JetGoodHiggs", pos=2),
        **jet_hists(coll="JetGoodHiggs", pos=3),
        # **parton_hists(coll="bQuarkHiggsMatched", pos=0),
        # **parton_hists(coll="bQuarkHiggsMatched", pos=1),
        # **parton_hists(coll="bQuarkHiggsMatched", pos=2),
        # **parton_hists(coll="bQuarkHiggsMatched", pos=3),
        # **parton_hists(coll="bQuarkHiggsMatched"),
        # **parton_hists(coll="JetGoodHiggsMatched", pos=0),
        # **parton_hists(coll="JetGoodHiggsMatched", pos=1),
        # **parton_hists(coll="JetGoodHiggsMatched", pos=2),
        # **parton_hists(coll="JetGoodHiggsMatched", pos=3),
        # **parton_hists(coll="bQuarkMatched", pos=0),
        # **parton_hists(coll="bQuarkMatched", pos=1),
        # **parton_hists(coll="bQuarkMatched", pos=2),
        # **parton_hists(coll="bQuarkMatched", pos=3),
        # **parton_hists(coll="bQuarkMatched"),
        # **parton_hists(coll="JetGoodMatched", pos=0),
        # **parton_hists(coll="JetGoodMatched", pos=1),
        # **parton_hists(coll="JetGoodMatched", pos=2),
        # **parton_hists(coll="JetGoodMatched", pos=3),
        # **parton_hists(coll="JetGoodMatched"),
        }


variables_dict_higgs_mass={
        "RecoHiggs1Mass": HistConf(
            [
                Axis(
                    coll=f"HiggsLeading",
                    field="mass",
                    bins=240,
                    start=0,
                    stop=240,
                    label=r"$M_{H_1}$ SPANet",
                )
            ],

        ),
        "RecoHiggs1Mass_Dhh": HistConf(
            [
                Axis(
                    coll=f"HiggsLeadingRun2",
                    field="mass",
                    bins=240,
                    start=0,
                    stop=240,
                    label=r"$M_{H_1}$ $D_{HH}$",
                )
            ],
        ),
        "RecoHiggs2Mass": HistConf(
            [
                Axis(
                    coll=f"HiggsSubLeading",
                    field="mass",
                    bins=240,
                    start=0,
                    stop=240,
                    label=r"$M_{H_2}$ SPANet",
                )
            ],
        ),
        "RecoHiggs2Mass_Dhh": HistConf(
            [
                Axis(
                    coll=f"HiggsSubLeadingRun2",
                    field="mass",
                    bins=240,
                    start=0,
                    stop=240,
                    label=r"$M_{H_2}$ $D_{HH}$",
                )
            ],
        )
    }


variables_dict_random_pt = {
        "Random_pt_Factor": HistConf(
            [
                Axis(
                    coll=f"events",
                    field="random_pt_weights",
                    bins=50,
                    start=0,
                    stop=2,
                    label=r"$pT$",
                )
            ],
        ),
    }

variables_dict_vbf={
        # **count_hist(coll="JetGood", bins=10, start=0, stop=10),
        # **jet_hists_dict(coll="JetGood", start=1, end=5),
        # **create_HistConf("JetGoodVBF", "eta", bins=60, start=-5, stop=5, label="JetGoodVBFeta"),
        # **create_HistConf("JetGoodVBF", "btagPNetQvG", pos=0, bins=60, start=0, stop=1, label="JetGoodVBFQvG_0"),
        # **create_HistConf("JetGoodVBF", "btagPNetQvG", pos=1, bins=60, start=0, stop=1, label="JetGoodVBFQvG_1"),
        # **create_HistConf("events", "deltaEta", bins=60, start=5, stop=10, label="JetGoodVBFdeltaEta"),
        # **create_HistConf("JetVBF_generalSelection", "eta", bins=60, start=-5, stop=5, label="JetVBFgeneralSelectionEta"),
        # **create_HistConf("JetVBF_generalSelection", "btagPNetQvG", pos=0, bins=60, start=0, stop=1, label="JetVBFgeneralSelectionQvG_0"),
        # **create_HistConf("JetVBF_generalSelection", "btagPNetQvG", pos=1, bins=60, start=0, stop=1, label="JetVBFgeneralSelectionQvG_1"),
        #
        #
        # **create_HistConf(
        #     "JetVBF_matched",
        #     "eta",
        #     bins=60,
        #     start=-5,
        #     stop=5,
        #     label="JetVBF_matched_eta",
        # ),
        # **create_HistConf(
        #     "events",
        #     "etaProduct",
        #     bins=5,
        #     start=-2.5,
        #     stop=2.5,
        #     label="JetVBF_matched_eta_product",
        # ),
        # **create_HistConf(
        #     "JetVBF_matched",
        #     "pt",
        #     bins=100,
        #     start=0,
        #     stop=1000,
        #     label="JetVBF_matched_pt",
        # ),
        # **create_HistConf(
        #     "JetVBF_matched",
        #     "btagPNetQvG",
        #     pos=0,
        #     bins=60,
        #     start=0,
        #     stop=1,
        #     label="JetVBF_matchedQvG_0",
        # ),
        # **create_HistConf(
        #     "JetVBF_matched",
        #     "btagPNetQvG",
        #     pos=1,
        #     bins=60,
        #     start=0,
        #     stop=1,
        #     label="JetVBF_matchedQvG_1",
        # ),
        # **create_HistConf(
        #     "quarkVBF_matched",
        #     "eta",
        #     bins=60,
        #     start=-5,
        #     stop=5,
        #     label="quarkVBF_matched_Eta",
        # ),
        # **create_HistConf(
        #     "quarkVBF_matched",
        #     "pt",
        #     bins=100,
        #     start=0,
        #     stop=1000,
        #     label="quarkVBF_matched_pt",
        # ),
        # **create_HistConf(
        #     "JetVBF_matched",
        #     "btagPNetB",
        #     bins=100,
        #     start=0,
        #     stop=1,
        #     label="JetGoodVBF_matched_btag",
        # ),
        # **create_HistConf(
        #     "events", "deltaEta_matched", bins=100, start=0, stop=10, label="deltaEta"
        # ),
        # **create_HistConf(
        #     "events", "jj_mass_matched", bins=100, start=0, stop=5000, label="jj_mass"
        # ),
        # **create_HistConf("HH", "mass", bins=100, start=0, stop=2500, label="HH_mass"),


        # variables from renato
        # **create_HistConf("events", "HH_deltaR", bins=50, start=0, stop=8, label="HH_deltaR"),
        # **create_HistConf("events", "H1j1_deltaR", bins=50, start=0, stop=8, label="H1j1_deltaR"),
        # **create_HistConf("events", "H1j2_deltaR", bins=50, start=0, stop=8, label="H1j2_deltaR"),
        # **create_HistConf("events", "H2j1_deltaR", bins=50, start=0, stop=8, label="H2j1_deltaR"),
        # **create_HistConf("events", "HH_centrality", bins=50, start=0, stop=1, label="HH_centrality"),

        # **create_HistConf("HH", "pt", bins=100, start=0, stop=800, label="HH_pt"),
        # **create_HistConf("HH", "eta", bins=60, start=-6, stop=6, label="HH_eta"),
        # **create_HistConf("HH", "phi", bins=60, start=-5, stop=5, label="HH_phi"),
        # **create_HistConf("HH", "mass", bins=100, start=0, stop=2200, label="HH_mass"),

        # **create_HistConf("HiggsLeading", "pt", bins=100, start=0, stop=800, label="HiggsLeading_pt"),
        # **create_HistConf("HiggsLeading", "eta", bins=60, start=-5, stop=5, label="HiggsLeading_eta"),
        # **create_HistConf("HiggsLeading", "phi", bins=60, start=-5, stop=5, label="HiggsLeading_phi"),
        # **create_HistConf("HiggsLeading", "mass", bins=100, start=0, stop=500, label="HiggsLeading_mass"),

        # **create_HistConf("HiggsSubLeading", "pt", bins=100, start=0, stop=800, label="HiggsSubLeading_pt"),
        # **create_HistConf("HiggsSubLeading", "eta", bins=60, start=-5, stop=5, label="HiggsSubLeading_eta"),
        # **create_HistConf("HiggsSubLeading", "phi", bins=60, start=-5, stop=5, label="HiggsSubLeading_phi"),
        # **create_HistConf("HiggsSubLeading", "mass", bins=100, start=0, stop=500, label="HiggsSubLeading_mass"),

        # **create_HistConf("Jet", "pt", bins=100, pos=0, start=0, stop=800, label="Jet_pt0"),
        # **create_HistConf("Jet", "pt", bins=100, pos=1, start=0, stop=800, label="Jet_pt1"),
        # **create_HistConf("Jet", "eta", bins=60, pos=0, start=-5, stop=5, label="Jet_eta0"),
        # **create_HistConf("Jet", "eta", bins=60, pos=1, start=-5, stop=5, label="Jet_eta1"),
        # **create_HistConf("Jet", "phi", bins=60, pos=0, start=-5, stop=5, label="Jet_phi0"),
        # **create_HistConf("Jet", "phi", bins=60, pos=1, start=-5, stop=5, label="Jet_phi1"),
        # **create_HistConf("Jet", "mass", bins=100, pos=0, start=0, stop=150, label="Jet_mass0"),
        # **create_HistConf("Jet", "mass", bins=100, pos=1, start=0, stop=150, label="Jet_mass1"),
        # **create_HistConf("Jet", "btagPNetB", pos=0, bins=100, start=0, stop=1, label="Jet_btagPNetB0"),
        # **create_HistConf("Jet", "btagPNetB", pos=1, bins=100, start=0, stop=1, label="Jet_btagPNetB1"),
        # **create_HistConf("Jet", "btagPNetQvG", pos=0, bins=100, start=0, stop=1, label="Jet_btagPNetQvG0"),
        # **create_HistConf("Jet", "btagPNetQvG", pos=1, bins=100, start=0, stop=1, label="Jet_btagPNetQvG1"),

        # **create_HistConf("JetGoodFromHiggsOrdered", "pt", bins=100, pos=0, start=0, stop=700, label="JetGoodFromHiggsOrdered_pt0"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "eta", bins=60, pos=0, start=-5, stop=5, label="JetGoodFromHiggsOrdered_eta0"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "phi", bins=60, pos=0, start=-5, stop=5, label="JetGoodFromHiggsOrdered_phi0"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "mass", bins=100, pos=0, start=0, stop=80, label="JetGoodFromHiggsOrdered_mass0"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetB", bins=100, pos=0, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetB0"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetQvG", bins=100, pos=0, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetQvG0"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "pt", bins=100, pos=1, start=0, stop=700, label="JetGoodFromHiggsOrdered_pt1"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "eta", bins=60, pos=1, start=-5, stop=5, label="JetGoodFromHiggsOrdered_eta1"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "phi", bins=60, pos=1, start=-5, stop=5, label="JetGoodFromHiggsOrdered_phi1"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "mass", bins=100, pos=1, start=0, stop=80, label="JetGoodFromHiggsOrdered_mass1"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetB", bins=100, pos=1, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetB1"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetQvG", bins=100, pos=1, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetQvG1"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "pt", bins=100, pos=2, start=0, stop=700, label="JetGoodFromHiggsOrdered_pt2"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "eta", bins=60, pos=2, start=-5, stop=5, label="JetGoodFromHiggsOrdered_eta2"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "phi", bins=60, pos=2, start=-5, stop=5, label="JetGoodFromHiggsOrdered_phi2"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "mass", bins=100, pos=2, start=0, stop=80, label="JetGoodFromHiggsOrdered_mass2"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetB", bins=100, pos=2, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetB2"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetQvG", bins=100, pos=2, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetQvG2"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "pt", bins=100, pos=3, start=0, stop=700, label="JetGoodFromHiggsOrdered_pt3"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "eta", bins=60, pos=3, start=-5, stop=5, label="JetGoodFromHiggsOrdered_eta3"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "phi", bins=60, pos=3, start=-5, stop=5, label="JetGoodFromHiggsOrdered_phi3"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "mass", bins=100, pos=3, start=0, stop=80, label="JetGoodFromHiggsOrdered_mass3"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetB", bins=100, pos=3, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetB3"),
        # **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetQvG", bins=100, pos=3, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetQvG3"),

        # **create_HistConf("JetVBFLeadingPtNotFromHiggs", "pt", bins=100, pos=0, start=0, stop=700, label="JetVBFLeadingPtNotFromHiggs_pt0"),
        # **create_HistConf("JetVBFLeadingPtNotFromHiggs", "eta", bins=60, pos=0, start=-5, stop=5, label="JetVBFLeadingPtNotFromHiggs_eta0"),
        # **create_HistConf("JetVBFLeadingPtNotFromHiggs", "phi", bins=60, pos=0, start=-5, stop=5, label="JetVBFLeadingPtNotFromHiggs_phi0"),
        # **create_HistConf("JetVBFLeadingPtNotFromHiggs", "mass", bins=100, pos=0, start=0, stop=75, label="JetVBFLeadingPtNotFromHiggs_mass0"),
        # **create_HistConf("JetVBFLeadingPtNotFromHiggs", "btagPNetB", bins=100, pos=0, start=0, stop=1, label="JetVBFLeadingPtNotFromHiggs_btagPNetB0"),
        # **create_HistConf("JetVBFLeadingPtNotFromHiggs", "btagPNetQvG", bins=100, pos=0, start=0, stop=1, label="JetVBFLeadingPtNotFromHiggs_btagPNetQvG0"),
        # **create_HistConf("JetVBFLeadingPtNotFromHiggs", "pt", bins=100, pos=1, start=0, stop=700, label="JetVBFLeadingPtNotFromHiggs_pt1"),
        # **create_HistConf("JetVBFLeadingPtNotFromHiggs", "eta", bins=60, pos=1, start=-5, stop=5, label="JetVBFLeadingPtNotFromHiggs_eta1"),
        # **create_HistConf("JetVBFLeadingPtNotFromHiggs", "phi", bins=60, pos=1, start=-5, stop=5, label="JetVBFLeadingPtNotFromHiggs_phi1"),
        # **create_HistConf("JetVBFLeadingPtNotFromHiggs", "mass", bins=100, pos=1, start=0, stop=75, label="JetVBFLeadingPtNotFromHiggs_mass1"),
        # **create_HistConf("JetVBFLeadingPtNotFromHiggs", "btagPNetB", bins=100, pos=1, start=0, stop=1, label="JetVBFLeadingPtNotFromHiggs_btagPNetB1"),
        # **create_HistConf("JetVBFLeadingPtNotFromHiggs", "btagPNetQvG", bins=100, pos=1, start=0, stop=1, label="JetVBFLeadingPtNotFromHiggs_btagPNetQvG1"),

        # **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "pt", bins=100, pos=0, start=0, stop=700, label="JetVBFLeadingMjjNotFromHiggs_pt0"),
        # **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "eta", bins=60, pos=0, start=-5, stop=5, label="JetVBFLeadingMjjNotFromHiggs_eta0"),
        # **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "phi", bins=60, pos=0, start=-5, stop=5, label="JetVBFLeadingMjjNotFromHiggs_phi0"),
        # **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "mass", bins=100, pos=0, start=0, stop=75, label="JetVBFLeadingMjjNotFromHiggs_mass0"),
        # **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "btagPNetB", bins=100, pos=0, start=0, stop=1, label="JetVBFLeadingMjjNotFromHiggs_btagPNetB0"),
        # **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "btagPNetQvG", bins=100, pos=0, start=0, stop=1, label="JetVBFLeadingMjjNotFromHiggs_btagPNetQvG0"),
        # **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "pt", bins=100, pos=1, start=0, stop=700, label="JetVBFLeadingMjjNotFromHiggs_pt1"),
        # **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "eta", bins=60, pos=1, start=-5, stop=5, label="JetVBFLeadingMjjNotFromHiggs_eta1"),
        # **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "phi", bins=60, pos=1, start=-5, stop=5, label="JetVBFLeadingMjjNotFromHiggs_phi1"),
        # **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "mass", bins=100, pos=1, start=0, stop=75, label="JetVBFLeadingMjjNotFromHiggs_mass1"),
        # **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "btagPNetB", bins=100, pos=1, start=0, stop=1, label="JetVBFLeadingMjjNotFromHiggs_btagPNetB1"),
        # **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "btagPNetQvG", bins=100, pos=1, start=0, stop=1, label="JetVBFLeadingMjjNotFromHiggs_btagPNetQvG1"),

        # **create_HistConf("events", "JetVBFLeadingPtNotFromHiggs_deltaEta", bins=11, start=0, stop=10, label="JetVBFLeadingPtNotFromHiggs_deltaEta"),
        # **create_HistConf("events", "JetVBFLeadingMjjNotFromHiggs_deltaEta", bins=11, start=0, stop=10, label="JetVBFLeadingMjjNotFromHiggs_deltaEta"),
        # **create_HistConf("events", "JetVBFLeadingPtNotFromHiggs_jjMass", bins=100, start=0, stop=2000, label="JetVBFLeadingPtNotFromHiggs_jjMass"),
        # **create_HistConf("events", "JetVBFLeadingMjjNotFromHiggs_jjMass", bins=100, start=0, stop=2000, label="JetVBFLeadingMjjNotFromHiggs_jjMass"),
}

def get_variables_dict(CLASSIFICATION=False, RANDOM_PT=False, VBF_VARIABLES=False):
    if CLASSIFICATION:
        variables_dict.update(variables_dict_higgs_mass)
    if RANDOM_PT:
        variables_dict.update(variables_dict_random_pt)


DEFAULT_COLLECTION_COLUMNS = ["JetGoodMatched", "JetGoodHiggsMatched", "JetGood", "JetGoodHiggs"]
DEFAULT_COLUMN_PARAMS = [["provenance", "pt", "eta", "phi", "mass", "btagPNetB"] for x in range(len(DEFAULT_COLLECTION))]
DEFAULT_EVENT_COLS = []


def get_columns_list(collection_columns=DEFAULT_COLLECTION_COLUMNS, column_parameters=DEFAULT_COLUMN_PARAMS, event_columns=DEFAULT_EVENT_COLS, flatten=False):
    ''' Function to create the column definition for the PocketCoffea Configurator().
    If any of the input options is set to `None`, the default option is used. To not save anything, use `[]`.

    :param: collection_columns : The collections that are to be saved. 
    :param: column_parameters  : The parameters for each collection to be saved. Should be a list of lists.
                                 One list for each collection.
    :param: event_columns      : parameters to be saved in the `event` section / Event-level parameters
    '''
    assert len(collection_columns) == len(column_parameters)
    if not collection_columns is list and not column_parameters is list and not event_columns is list:
        raise TypeError("Inputs must be lists.")
    columns = []
    for collection, params in zip(collection_columns, column_parameters):
        columns.append(ColOut(collection, params, flatten))
    columns.append("events",event_columns,flatten)
    return columns







