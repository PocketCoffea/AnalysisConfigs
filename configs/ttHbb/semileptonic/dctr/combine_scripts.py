import os
from pocket_coffea.utils.stat.combine import Datacard

def create_scripts(
    datacards: dict[Datacard],
    directory: str,
    card_name: str = "datacard_combined.txt",
    workspace_name : str = "workspace.root",
    categories_masked : list[str] = None,
    suffix: str = None,
    ) -> None:
    """
    Write the bash scripts to run the fit with CMS Combine Tool."""

    # Save fit scripts
    freezeParameters = ["r"]

    args = {
        "run_MultiDimFit.sh": [[
            "combine -M MultiDimFit",
            f"-d {workspace_name}",
            "-n .snapshot_all_channels",
            f"--freezeParameters {','.join(freezeParameters)}",
            "--cminDefaultMinimizerStrategy 2 --robustFit=1",
            "--saveWorkspace",
        ]],
        "run_FitDiagnostics.sh": [[
            "combine -M FitDiagnostics",
            f"-d {workspace_name}",
            "-n .snapshot_all_channels",
            f"--freezeParameters {','.join(freezeParameters)}",
            "--cminDefaultMinimizerStrategy 2 --robustFit=1",
            "--saveWorkspace",
            "--saveShapes",
            "--saveWithUncertainties"
        ]],
        "run_MultiDimFit_scan1d.sh": [[
            "combine -M MultiDimFit",
            f"-d {workspace_name}",
            "-n .scan1d",
            f"--freezeParameters {','.join(freezeParameters)}",
            "-t -1 --toysFrequentist --expectSignal=1",
            "--cminDefaultMinimizerStrategy 2 --robustFit=1",
            "--saveWorkspace",
            "-v 2 --algo grid --points=30 --rMin 0 --rMax 2",
        ]],
        "run_MultiDimFit_toysFrequentist.sh": [[
            "combine -M MultiDimFit",
            f"-d {workspace_name}",
            "-n .asimov_fit",
            "-t -1 --toysFrequentist --expectSignal=1",
            "--cminDefaultMinimizerStrategy 2 --robustFit=1",
            "--saveWorkspace -v 2",
        ]],
        "run_FitDiagnostics_toysFrequentist.sh": [[
            "combine -M FitDiagnostics",
            f"-d {workspace_name}",
            "-n .asimov_fit",
            "-t -1 --toysFrequentist --expectSignal=1",
            "--cminDefaultMinimizerStrategy 2 --robustFit=1",
            "--saveWorkspace -v 2",
        ]],
        "run_MultiDimFit_toysFrequentist_scan1d.sh": [[
            "combine -M MultiDimFit",
            f"-d {workspace_name}",
            "-n .asimov_scan1d",
            "-t -1 --toysFrequentist --expectSignal=1",
            "--cminDefaultMinimizerStrategy 2 --robustFit=1",
            "--saveWorkspace",
            "-v 2 --algo grid --points=30 --rMin 0 --rMax 2",
        ]],
        "run_impacts.sh": [
            [f"combineTool.py -M Impacts -d {workspace_name}",
            f"--freezeParameters {','.join(freezeParameters)}",
            "-t -1 --toysFrequentist --expectSignal=1 --cminDefaultMinimizerStrategy 2 --robustFit=1",
            "-v 2 --rMin 0 --rMax 2 -m 125 --doInitialFit"],
            [f"combineTool.py -M Impacts -d {workspace_name}",
            f"--freezeParameters {','.join(freezeParameters)}",
            "-t -1 --toysFrequentist --expectSignal=1 --cminDefaultMinimizerStrategy 2 --robustFit=1",
            "-v 2 --rMin 0 --rMax 2 -m 125 --doFits --job-mode slurm --job-dir jobs --parallel 100"]
        ],
        "plot_impacts.sh": [
            [f"combineTool.py -M Impacts -d {workspace_name}",
            f"--freezeParameters {','.join(freezeParameters)}",
            "-t -1 --toysFrequentist --expectSignal=1 --cminDefaultMinimizerStrategy 2 --robustFit=1",
            "-v 2 --rMin 0 --rMax 2 -m 125 -o impacts.json"],
            ["plotImpacts.py -i impacts.json -o impacts"]
        ],
        # -v 2 --rMin -5 --rMax 5 --robustHesse=1 --robustHesseSave 1 --saveFitResult
        "run_correlation_matrix.sh": [
            ["combine -M MultiDimFit",
            f"-d {workspace_name}",
            "-n .covariance_matrix",
            "-t -1 --toysFrequentist --expectSignal=1",
            "--cminDefaultMinimizerStrategy 2 --robustFit=1",
            "-v 2 --rMin -5 --rMax 5 --robustHesse=1 --robustHesseSave 1 --saveFitResult"]
        ]
    }
    if categories_masked:
        args.update({
            f"run_MultiDimFit_mask_{'_'.join(categories_masked)}.sh" : [[
                "combine -M MultiDimFit",
                f"-d {workspace_name}",
                f"-n .snapshot_{'_'.join(categories_masked)}",
                f"--freezeParameters {','.join(freezeParameters)}",
                "--cminDefaultMinimizerStrategy 2 --robustFit=1",
                "--saveWorkspace",
                f"--setParameters {','.join([f'mask_{cat}=1' for cat in categories_masked ])}"
            ]],
            f"run_FitDiagnostics_mask_{'_'.join(categories_masked)}.sh" : [[
                "combine -M FitDiagnostics",
                f"-d {workspace_name}",
                f"-n .snapshot_{'_'.join(categories_masked)}",
                f"--freezeParameters {','.join(freezeParameters)}",
                "--cminDefaultMinimizerStrategy 2 --robustFit=1",
                "--saveWorkspace",
                "--saveShapes",
                "--saveWithUncertainties",
                f"--setParameters {','.join([f'mask_{cat}=1' for cat in categories_masked ])}"
            ]],
        })

    scripts = {}
    for path, lines in args.items():
        scripts[path] = [f"{' '.join(l)}\n" for l in lines]

    for script_name, commands in scripts.items():
        script_name = script_name.replace(".sh", f"_{suffix}.sh") if suffix else script_name
        output_file = os.path.join(directory, script_name)
        print(f"Writing fit script to {output_file}")
        with open(output_file, "w") as file:
            file.write("#!/bin/bash\n")
            file.writelines(commands)