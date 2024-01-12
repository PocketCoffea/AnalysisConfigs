# Cross-section for inclusive ttbar production in pb [https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO] (updated July 2022)
# Cross-section for inclusive ttH production in pb, computed at NNLO [https://arxiv.org/pdf/2210.07846.pdf]
xsec_ttbar = 833.9
xsec_tth = 0.5070

# Branching ratio of Higgs to bbbar [https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR]
br_H_bb = 0.5824

# Cross-section for inclusive ttHbb production in pb
xsec_tthbb = xsec_tth * br_H_bb

# Branching ratio for W boson decaying to leptons (BR(W->l nu) = 0.1086)
br_W_lep = 0.1086
br_W_lep_any = 3 * br_W_lep

# Branching ratio for W boson decaying to hadrons (BR(W->qq) = 0.6741)
br_W_had = 0.6741

# Branching ratio for the semileptonic decay of the top pair (BR(tt->WbWb->l nu b q q b) = 2 * 3 * BR(W->l nu) * BR(W->qq) = 0.1086 * 0.6741 * 2 * 3 = 0.463)
# A factor 2 takes into account the fact that the leptonic decay of the W boson can happen in either the first or the second top quark
br_semilep = 2 * br_W_lep_any * br_W_had

# Cross-section for the semileptonic decay of the top pair in pb
xsec_ttbar_semilep = xsec_ttbar * br_semilep
xsec_tthbb_semilep = xsec_tthbb * br_semilep

# Cross-section for the dileptonic decay of the top pair in pb
xsec_ttbar_dilep = xsec_ttbar * (br_W_lep_any ** 2)
xsec_tthbb_dilep = xsec_tthbb * (br_W_lep_any ** 2)

# Cross-section for the hadronic decay of the top pair in pb
xsec_ttbar_had = xsec_ttbar * (br_W_had ** 2)
xsec_tthbb_had = xsec_tthbb * (br_W_had ** 2)

print(f"Inclusive ttbar cross-section: {xsec_ttbar} pb")
print("Branching ratios:")
print(f"BR(W->l nu) = {br_W_lep}")
print(f"BR(W->l nu) * 3 = {br_W_lep_any}")
print(f"BR(W->qq) = {br_W_had}")
print(f"BR(tt->WbWb->l nu b q q b) = {br_semilep}")
print(f"Semileptonic ttbar cross-section: {xsec_ttbar_semilep} pb")
print(f"Dileptonic ttbar cross-section: {xsec_ttbar_dilep} pb")
print(f"Hadronic ttbar cross-section: {xsec_ttbar_had} pb")
print("\n********************************************\n")
print(f"Inclusive ttH cross-section: {xsec_tth} pb")
print(f"ttHbb cross-section: {xsec_tthbb} pb")
