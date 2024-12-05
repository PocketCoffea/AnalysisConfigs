import argparse
from coffea.util import load
from pocket_coffea.utils.datacard import Datacard
from pocket_coffea.utils.processes import Process
from pocket_coffea.utils.systematics import SystematicUncertainty

parser = argparse.ArgumentParser(description="Make datacards")
parser.add_argument("-i", "--input", help="Coffea input file with histograms", required=True)
parser.add_argument("-o", "--output", help="Output directory for datacards", default="datacards", required=False)
parser.add_argument("--histogram", help="Histogram to make datacard for", default="spanet_tthbb_transformed",  required=False)
args = parser.parse_args()

df = load(args.input)
histograms = df["variables"][args.histogram]
datasets_metadata = df["datasets_metadata"]

processes = [
    Process(name="ttlf", samples=["TTToSemiLeptonic__TTToSemiLeptonic_tt+LF"], is_signal=False),
    Process(name="ttcc", samples=["TTToSemiLeptonic__TTToSemiLeptonic_tt+C"], is_signal=False),
    Process(name="ttbb", samples=["TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B"], is_signal=False),
    Process(name="singletop", samples=["SingleTop"], is_signal=False),
    Process(name="signal", samples=["ttHTobb"], is_signal=True),
]

systematics = [
    SystematicUncertainty(name="lumi", typ="lnN", processes=["ttlf", "ttcc", "ttbb", "signal"], value=0.9),
]

datacard = Datacard(
    histograms=histograms,
    datasets_metadata=datasets_metadata,
    processes=processes,
    systematics=systematics,
    year="2018",
    category="CR",
)
datacard.dump(directory=args.output, card_name="datacard.txt", shapes_filename="shapes.root")
