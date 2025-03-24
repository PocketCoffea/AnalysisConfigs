import os
import argparse
from coffea.util import load
from pocket_coffea.utils.stat.combine import Datacard, combine_datacards, create_scripts
from pocket_coffea.utils.stat.processes import Process
from pocket_coffea.utils.stat.systematics import SystematicUncertainty
from combine_scripts import create_scripts

parser = argparse.ArgumentParser(description="Make datacards")
parser.add_argument("-i", "--input", help="Coffea input file with histograms", required=True)
parser.add_argument("-o", "--output", help="Output directory for datacards", default="datacards", required=False)
parser.add_argument("--channel-masks", help="Create datacards with channel masks", action="store_true")
args = parser.parse_args()

df = load(args.input)
datasets_metadata = df["datasets_metadata"]

datacards_4j = {}

output_dir_4j = os.path.join(args.output, "fit_4j")

histograms_dict = {
    "CR" : df["variables"]["nJets"],
    "CR_ttlf_0p60" : df["variables"]["nJets"],
    "CR_ttcc" : df["variables"]["nLeptons"],
    "SR" : df["variables"]["spanet_tthbb_transformed_binning0p0125"],
}

bins_edges_dict = {
    "CR" : None,
    "CR_ttlf_0p60" : [4,5,6,7,14],
    "CR_ttcc" : None,
    "SR" : None,
}

for year in ["2016_PreVFP", "2016_PostVFP", "2017", "2018"]:
    print("Creating datacards for year: ", year)
    years = [year]
    label = year
    datacards_4j[year] = {}

    samples_data = set([s for year in years for s in datasets_metadata["by_datataking_period"][year].keys() if s.startswith("DATA")])

    processes = [
        Process(name="tthbb", samples=["ttHTobb"], years=years, is_signal=True),
        Process(name="ttlf", samples=["TTToSemiLeptonic__TTToSemiLeptonic_tt+LF"], years=years, is_signal=False),
        Process(name="ttcc", samples=["TTToSemiLeptonic__TTToSemiLeptonic_tt+C"], years=years, is_signal=False),
        Process(name="tt_dilepton", samples=["TTTo2L2Nu"], years=years, is_signal=False, has_rateParam=False),
        Process(name="singletop", samples=["SingleTop"], years=years, is_signal=False, has_rateParam=False),
        Process(name="vjets", samples=["WJetsToLNu_HT", "DYJetsToLL"], years=years, is_signal=False, has_rateParam=False),
        Process(name="ttv", samples=["TTV"], years=years, is_signal=False, has_rateParam=False),
        Process(name="diboson", samples=["VV"], years=years, is_signal=False, has_rateParam=False),
        Process(name="ttbb", samples=["TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B"], years=years, is_signal=False),
    ]

    processes_ttbb = [p.name for p in processes if p.name.startswith("ttbb")]

    data_processes = [Process(name="data_obs", samples=samples_data, years=[None], is_signal=False, is_data=True)]

    systematics = [
        SystematicUncertainty(name="lumi_2018", typ="lnN", processes=[p.name for p in processes], years=["2018"], value=1.025),
        SystematicUncertainty(name="lumi_2017", typ="lnN", processes=[p.name for p in processes], years=["2017"], value=1.023),
        SystematicUncertainty(name="lumi_2016", typ="lnN", processes=[p.name for p in processes], years=["2016_PreVFP", "2016_PostVFP"], value=1.012),
    ]
    common_systematics = [
        "pileup", "sf_jet_puId",
        "sf_ele_reco", "sf_ele_id",
        "sf_ele_trigger_era", "sf_ele_trigger_ht",
        "sf_ele_trigger_pileup", "sf_ele_trigger_stat",
        "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
        "sf_L1prefiring_no2018",
        "sf_btag_withcalib_complete_ttsplit_cferr1", "sf_btag_withcalib_complete_ttsplit_cferr2",
        "sf_btag_withcalib_complete_ttsplit_hf",
        "sf_btag_withcalib_complete_ttsplit_hfstats1", "sf_btag_withcalib_complete_ttsplit_hfstats2",
        "sf_btag_withcalib_complete_ttsplit_lf",
        "sf_btag_withcalib_complete_ttsplit_lfstats1", "sf_btag_withcalib_complete_ttsplit_lfstats2",
        "sf_ttlf_calib_with_ttcc_variations_norm_ttcc",
        "JES_Total_AK4PFchs", "JER_AK4PFchs"
    ]
    qcd_systematics = [
        "sf_qcd_renorm_scale",
        "sf_qcd_factor_scale"
    ]
    pdf_systematics = [
        "sf_lhe_pdf_weight"
    ]
    parton_shower_systematics = [
        "sf_partonshower_isr",
        "sf_partonshower_fsr"
    ]

    processes_without_qcd = ["diboson"]
    processes_without_pdf = ["diboson", "singletop"]
    for syst in common_systematics:
        for year in years:
            systematics.append(SystematicUncertainty(name=syst, datacard_name=f"{syst}_{year}", typ="shape", processes=[p.name for p in processes], years=[year], value=1.0))
    # Decorrelate QCD systematics across processes
    for syst in qcd_systematics:
        for process in processes:
            if process.name not in processes_without_qcd and process.name not in processes_ttbb:
                systematics.append(SystematicUncertainty(name=syst, datacard_name=f"{syst}_{process.name}", typ="shape", processes=[process.name], years=years, value=1.0))

    # Decorrelate PDF systematics across processes
    for syst in pdf_systematics:
        for process in processes:
            if process.name not in processes_without_pdf and process.name not in processes_ttbb:
                systematics.append(SystematicUncertainty(name=syst, datacard_name=f"{syst}_{process.name}", typ="shape", processes=[process.name], years=years, value=1.0))

    for syst in parton_shower_systematics:
        for process in processes:
            if process.name not in processes_ttbb:
                systematics.append(SystematicUncertainty(name=syst, datacard_name=f"{syst}_{process.name}", typ="shape", processes=[process.name], years=years, value=1.0))

    for syst in qcd_systematics + pdf_systematics + parton_shower_systematics:
        systematics.append(SystematicUncertainty(name=syst, datacard_name=f"{syst}_ttbb", typ="shape", processes=[p for p in processes_ttbb], years=years, value=1.0))

    # Add a lnN systematic uncertainty for the pdf weight of the singletop process
    systematics.append(SystematicUncertainty(name="sf_lhe_pdf_weight_singletop", typ="lnN", processes=["singletop"], years=years, value=1.05))

    print("Systematic lnN uncertainties: ", [s.name for s in systematics if s.typ == "lnN"])
    print("Systematic shape uncertainties: ", [s.name for s in systematics if s.typ == "shape"])

    for cat, histograms in histograms_dict.items():
        print(f"Creating datacard for category: {cat}")
        datacard = Datacard(
            histograms=histograms,
            datasets_metadata=datasets_metadata,
            cutflow=df["cutflow"],
            processes=processes,
            data_processes=data_processes,
            systematics=systematics,
            years=years,
            category=cat,
            bin_suffix=label,
            bins_edges=bins_edges_dict[cat]
        )
        datacard.dump(directory=output_dir_4j, card_name=f"datacard_{cat}_{label}.txt", shapes_filename=f"shapes_{cat}_{label}.root")
        datacards_4j[year][f'datacard_{cat}_{label}.txt'] = datacard
        print(f"Datacard saved in {os.path.abspath(os.path.join(output_dir_4j, f'datacard_{cat}_{label}.txt'))}")

    # Combine datacards, produce scripts with SR channel masked
    for _datacards, output_dir, categories_masked in zip([datacards_4j[year]], [output_dir_4j], [["SR"]]):
        datacard_name = f"datacard_combined_{label}.txt"
        workspace_name = f"workspace_{label}.root"
        combine_datacards(
            _datacards,
            directory=output_dir,
            path=f"combine_datacards_{label}.sh",
            card_name=datacard_name,
            workspace_name=workspace_name,
            channel_masks=False
        )

        create_scripts(
            _datacards,
            directory=output_dir,
            card_name=datacard_name,
            workspace_name=workspace_name,
            categories_masked=None
        )

        if args.channel_masks:
            datacard_name_mask = f"datacard_combined_{label}_mask.txt"
            workspace_name_mask = f"workspace_{label}_mask.root"
            combine_datacards(
                _datacards,
                directory=os.path.join(output_dir, "channel_masks"),
                path=f"combine_datacards_{label}_mask.sh",
                card_name=datacard_name_mask,
                workspace_name=workspace_name_mask,
                channel_masks=True
            )

            create_scripts(
                _datacards,
                directory=os.path.join(output_dir, "channel_masks"),
                card_name=datacard_name,
                workspace_name=workspace_name,
                categories_masked=categories_masked,
            )

# Combine datacards for all years
label = "run2"
datacard_name = f"datacard_combined_{label}.txt"
workspace_name = f"workspace_{label}.root"

for _datacards, output_dir, categories_masked in zip([datacards_4j], [output_dir_4j], [["SR"]]):
    combine_datacards(
        {filename : datacard for year in _datacards for filename, datacard in _datacards[year].items()},
        directory=output_dir,
        path=f"combine_datacards_{label}.sh",
        card_name=datacard_name,
        workspace_name=workspace_name,
        channel_masks=False
    )

    combine_datacards(
        {filename : datacard for year in _datacards for filename, datacard in _datacards[year].items()},
        directory=os.path.join(output_dir, "channel_masks"),
        path=f"combine_datacards_{label}.sh",
        card_name=datacard_name,
        workspace_name=workspace_name,
        channel_masks=True
    )

    create_scripts(
        {filename : datacard for year in _datacards for filename, datacard in _datacards[year].items()},
        directory=output_dir,
        card_name=datacard_name,
        workspace_name=workspace_name,
        categories_masked=None,
        suffix=label
    )

    create_scripts(
        {filename : datacard for year in _datacards for filename, datacard in _datacards[year].items()},
        directory=os.path.join(output_dir, "channel_masks"),
        card_name=datacard_name,
        workspace_name=workspace_name,
        categories_masked=categories_masked,
        suffix=label
    )