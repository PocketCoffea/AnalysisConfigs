from pocket_coffea.lib.cut_definition import Cut
import custom_cut_functions as cuts_f
from params.preselection import cuts

def get_trigger_passfail(triggers, category):
    return Cut(
        name=f"{'_'.join(triggers)}_{category}",
        params={"triggers": triggers, "category": category},
        function=cuts_f.trigger_mask
    )

def get_trigger_passfail_2017(triggers, category):
    return Cut(
        name=f"{'_'.join(triggers)}_{category}",
        params={"triggers": triggers, "category": category},
        function=cuts_f.trigger_mask_2017
    )

def get_ht_above(minht, name=None):
    if name == None:
        name = f"minht{minht}"
    return Cut(
        name=name,
        params={"minht": minht},
        function=cuts_f.ht_above
    )

def get_ht_below(maxht, name=None):
    if name == None:
        name = f"maxht{maxht}"
    return Cut(
        name=name,
        params={"maxht": maxht},
        function=cuts_f.ht_below
    )

semileptonic_presel_triggerSF = Cut(
    name="semileptonic_triggerSF",
    params=cuts,
    function=cuts_f.semileptonic_triggerSF,
)
