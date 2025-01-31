xsecs = {
    "DYJetsToLL_M-50_HT-70to100": 175.3,
    "DYJetsToLL_M-50_HT-100to200": 147.4,
    "DYJetsToLL_M-50_HT-200to400": 40.99,
    "DYJetsToLL_M-50_HT-400to600": 5.678,
    "DYJetsToLL_M-50_HT-600to800": 1.367,
    "DYJetsToLL_M-50_HT-800to1200": 0.6304,
    "DYJetsToLL_M-50_HT-1200to2500": 0.1514,
    "DYJetsToLL_M-50_HT-2500toInf": 0.003565,
}

kfactors = {
    "2016_PreVFP" : 1.227,
    "2016_PostVFP" : 1.227,
    "2017" : 1.137,
    "2018" : 1.137,
}

for dataset, xsec in xsecs.items():
    print (f"dataset: {dataset}, xsec: {xsec}")
    for year, kfactor in kfactors.items():
        print (f"year: {year}, xsec*kfactor: {xsec*kfactor}")
    print ("")
