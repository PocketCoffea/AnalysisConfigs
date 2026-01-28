import numpy as np
import awkward as ak
import correctionlib
from configs.ttHbb.semileptonic.common.cuts.custom_cut_functions import eq_genTtbarId_100
from collections import defaultdict
from pocket_coffea.lib.weights import WeightWrapper, WeightLambda, WeightData, WeightDataMultiVariation


def sf_btag_withcalibration_ttsplit(events, params, sample, jets, year, njets, jetsHt, variations=["central"]):
    '''
    DeepJet (or other taggers) AK4 btagging SF.
    See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/BTV_2018_UL_btagging.html
    The scale factors have 8 default uncertainty
    sources (hf,lf,hfstats1/2,lfstats1/2,cferr1/2) (all of this up_*var*, and down_*var*).
    All except the cferr1/2 uncertainties are to be
    applied to light and b jets. The cferr1/2 uncertainties are to be applied to c jets.
    hf/lfstats1/2 uncertainties are to be decorrelated between years, the others correlated.
    Additional jes-varied scale factors are supplied to be applied for the jes variations.

    if variation is not one of the jes ones both the up and down sf is returned.
    If variation is a jet variation the argument must be up_jes* or down_jes* since it is applied on the specified
    Jes variation jets.
    '''
    btagSF = params.jet_scale_factors.btagSF[year]
    btag_discriminator = params.btagging.working_point[year]["btagging_algorithm"]
    cset = correctionlib.CorrectionSet.from_file(btagSF.file)
    corr = cset[btagSF.name]

    ttbar_sample = False
    if sample in ["TTbbSemiLeptonic", "TTToSemiLeptonic"]:
        ttbar_sample = True
        cset_calib = correctionlib.CorrectionSet.from_file(
            params.btagSF_calibration_ttsplit[year]["file"]
        )
        ## CAREFUL: This is the calibration file specific for the ttbar split
        corr_calib = cset_calib[params.btagSF_calibration_ttsplit[year]["name"]]
    else:
        cset_calib = correctionlib.CorrectionSet.from_file(
            params.btagSF_calibration[year]["file"]
        )
        corr_calib = cset_calib[params.btagSF_calibration[year]["name"]]

    flavour = ak.to_numpy(ak.flatten(jets.hadronFlavour))
    abseta = np.abs(ak.to_numpy(ak.flatten(jets.eta)))
    pt = ak.to_numpy(ak.flatten(jets.pt))
    discr = ak.to_numpy(ak.flatten(jets[btag_discriminator]))
    njets = ak.to_numpy(njets)
    jetsHt = ak.to_numpy(jetsHt)

    central_SF_byjet = corr.evaluate("central", flavour, abseta, pt, discr)


    def _get_sf_variation_with_mask(variation, mask):
        index = (np.indices(discr.shape)).flatten()[mask]
        # Copying the central SF
        sf = np.copy(central_SF_byjet)
        w = corr.evaluate(variation, flavour[mask], abseta[mask], pt[mask], discr[mask])
        sf[index] = w
        sf_out = ak.prod(ak.unflatten(sf, njets), axis=1)
        return sf_out

    output = {}
    for variation in variations:
        if variation == "central":
            output[variation] = [ak.prod(ak.unflatten(central_SF_byjet, njets), axis=1)]
        else:
            # Nominal sf==1
            nominal = np.ones(ak.num(njets, axis=0))
            # Systematic variations
            if "cferr" in variation:
                # Computing the scale factor only on c-flavour jets
                c_mask = flavour == 4
                output[variation] = [
                    nominal,
                    _get_sf_variation_with_mask(f"up_{variation}", c_mask),
                    _get_sf_variation_with_mask(f"down_{variation}", c_mask),
                ]

            elif variation.startswith("JES") and "AK4" in variation:
                # We need to convert the name of the variation
                # from JES_VariationUp to  up_jesVariation
                if variation.startswith("JES_Total") and variation[-2:] == "Up":
                    btag_jes_var = "up_jes"
                elif variation.startswith("JES_Total") and variation[-4:] == "Down":
                    btag_jes_var = "down_jes"
                else:
                    # we need to remove the possible jet type
                    variation = variation.replace("_AK4PFchs", "")
                    variation = variation.replace("_AK4PFPuppi", "")
                    if variation[-2:] == "Up":
                        btag_jes_var = f"up_jes{variation[4:-2]}"
                    elif variation[-4:] == "Down":
                        btag_jes_var = f"down_jes{variation[4:-4]}"
                # This is a special case where a dedicate btagSF is computed for up and down Jes shape variations.
                # This is not an up/down variation, but a single modified SF.
                # N.B: It is a central SF
                notc_mask = flavour != 4
                output["central"] = [_get_sf_variation_with_mask(btag_jes_var, notc_mask)]
            else:
                # Computing the scale factor only NON c-flavour jets
                notc_mask = flavour != 4
                output[variation] = [
                    nominal,
                    _get_sf_variation_with_mask(f"up_{variation}", notc_mask),
                    _get_sf_variation_with_mask(f"down_{variation}", notc_mask),
                ]
    # now multiplying by the calibration
    output_final = {}

    if ttbar_sample:
        # need to split events in ttbar parts
        lf_mask = eq_genTtbarId_100(events, params={"genTtbarId": [0]}, year=year, sample=sample)
        cc_mask = eq_genTtbarId_100(events, params={"genTtbarId": [41, 42, 43, 44, 45, 46]}, year=year, sample=sample)
        bb_mask = eq_genTtbarId_100(events, params={"genTtbarId": [51, 52, 53, 54, 55, 56]}, year=year, sample=sample)

        output_by_flavour = defaultdict(dict)
        subsamples = ["tt+LF", "tt+C", "tt+B"]

        for var, sf in output.items():
            if var =="central":
                for subsample in subsamples:
                    output_by_flavour[subsample][var] = [sf[0]*corr_calib.evaluate(f"{sample}__{sample}_{subsample}",
                                                                "nominal",
                                                                njets, jetsHt)]

            else:
                for subsample in subsamples:
                    output_by_flavour[subsample][var] = [
                        sf[0], #nominal
                        sf[1] * corr_calib.evaluate(f"{sample}__{sample}_{subsample}", f"sf_btag_{var}Up",njets, jetsHt), #up
                        sf[2] * corr_calib.evaluate(f"{sample}__{sample}_{subsample}", f"sf_btag_{var}Down", njets, jetsHt) #down
                    ]

        # Now using the maskes
        for var in output:
            if var == "central":
                output_final[var] = [lf_mask*output_by_flavour["tt+LF"][var][0] + 
                                     cc_mask*output_by_flavour["tt+C"][var][0] + 
                                     bb_mask*output_by_flavour["tt+B"][var][0]]

            else:
                output_final[var] = [
                    sf[0], #nominal
                    lf_mask*output_by_flavour["tt+LF"][var][1] + cc_mask*output_by_flavour["tt+C"][var][1] + bb_mask*output_by_flavour["tt+B"][var][1],
                    lf_mask*output_by_flavour["tt+LF"][var][2] + cc_mask*output_by_flavour["tt+C"][var][2] + bb_mask*output_by_flavour["tt+B"][var][2]
                ]
    else:
        for var, sf in output.items():
            if var =="central":
                output_final[var] = [sf[0]*corr_calib.evaluate(sample,
                                                          "nominal",
                                                          njets, jetsHt)]
            else:
                output_final[var] = [
                    sf[0], #nominal
                    sf[1] * corr_calib.evaluate(sample, f"sf_btag_{var}Up",njets, jetsHt), #up
                    sf[2] * corr_calib.evaluate(sample, f"sf_btag_{var}Down", njets, jetsHt) #down
                ]

    return output_final

###############

class SF_btag_withcalib_complete_ttsplit(WeightWrapper):
    name = "sf_btag_withcalib_complete_ttsplit"
    has_variations = True

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        # Getting the variations from the parameters depending on the year
        self._variations = params.systematic_variations.weight_variations.sf_btag[metadata["year"]]
        self.jet_coll = params.jet_scale_factors.jet_collection.btag

    def compute(self, events, size, shape_variation):
        jetsHt = ak.sum(events[self.jet_coll].pt, axis=1)
        
        if shape_variation == "nominal":
            out = sf_btag_withcalibration_ttsplit(events, self._params,
                          sample=self._metadata["sample"], 
                          jets=events[self.jet_coll],
                          year=self._metadata["year"],
                          # Assuming n*JetCollection* is defined
                          njets=events[f"n{self.jet_coll}"],
                          jetsHt=jetsHt,
                          variations=["central"] + self._variations,
                          )
            # This is a dict with variation: [nom, up, down]
            return WeightDataMultiVariation(
                name = self.name,
                nominal = out["central"][0],
                variations = self._variations,
                up = [out[var][1] for var in self._variations],
                down = [out[var][2] for var in self._variations]
            )

        elif shape_variation.startswith("JES"):
            out = sf_btag_withcalibration_ttsplit(events, self._params,
                                sample=self._metadata["sample"],
                                jets=events[self.jet_coll],
                                year=self._metadata["year"],
                                # Assuming n*JetCollection* is defined
                                njets=events[f"n{self.jet_coll}"],
                                jetsHt=jetsHt,
                                variations=[shape_variation],
                                )
            return WeightData(
                name = self.name,
                nominal = out["central"][0]
                )       
            
        else:
            out = sf_btag_withcalibration_ttsplit(events, self._params,
                          sample=self._metadata["sample"],
                          jets=events[self.jet_coll],
                          year=self._metadata["year"],
                          # Assuming n*JetCollection* is defined
                          njets=events[f"n{self.jet_coll}"],
                          jetsHt=jetsHt,
                          variations=["central"],
                          )
            return WeightData(
                name = self.name,
                nominal = out["central"][0]
            )
