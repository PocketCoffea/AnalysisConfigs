from coffea.util import load
o = load("out_cartesian/output_all.coffea")

print(o["variables"].keys())
print("\n")
print(o["columns"].keys())
# print first value of dict
print(o["columns"]["QCD"]["QCD_PT-15to7000_PT-15to7000_2018"]["baseline"][list(o["columns"]["QCD"]["QCD_PT-15to7000_PT-15to7000_2018"]["baseline"].keys())[0]])