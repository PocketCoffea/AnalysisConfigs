from coffea.util import load
o = load("out_cartesian/output_all.coffea")

print(o["variables"].keys())
print("\n")
print(o["columns"].keys())