#!/bin/bash

pocket-coffea build-datasets --cfg datasets/datasets_definitions_diboson.json -o -rs 'T[12]_(IT|DE|BE|FR|CH|PL|UK)_\w+' -p 6 -bs T2_CH_CSCS,T3_CH_PSI,T2_DE_DESY
pocket-coffea build-datasets --cfg datasets/datasets_definitions.json -o -rs 'T[12]_(IT|DE|BE|FR|CH|PL|UK|ES)_\w+' -p 6 -bs T2_CH_CSCS,T3_CH_PSI,T2_DE_DESY
pocket-coffea build-datasets --cfg datasets/datasets_definitions_ttW.json -o -rs 'T[12]_(FR|IT|BE|CH|PL|UK)_\w+' -p 6 -bs T2_CH_CSCS,T3_CH_PSI,T2_DE_DESY
pocket-coffea build-datasets --cfg datasets/datasets_definitions_ttZToLLNuNu.json -o -rs 'T[12]_(FR|IT|BE|CH|PL|UK)_\w+' -p 6 -bs T2_CH_CSCS,T3_CH_PSI,T2_DE_DESY
pocket-coffea build-datasets --cfg datasets/datasets_definitions_ttZToQQ.json -o -rs 'T[12]_(IT|BE|FR|CH|PL|UK)_\w+' -p 6 -bs T2_CH_CSCS,T3_CH_PSI,T2_DE_DESY
pocket-coffea build-datasets --cfg datasets/datasets_definitions_ZJets.json -o -rs 'T[12]_(IT|DE|BE|FR|CH|PL|UK)_\w+' -p 6 -bs T2_CH_CSCS,T3_CH_PSI,T2_DE_DESY
