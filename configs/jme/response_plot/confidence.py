import numpy as np

# sigmas = [0.99, 0.98, 0.95, 0.87, 0.68]
# conversions = [2.576,2.326,1.960,1.514,0.9945]
# for perc, conv in zip(sigmas,conversions):
#    width = Confidence(hist, confLevel = perc) # /(2* conv)
# 99% confidence is [-2.5sigma,+2.5sigma] so divide by 2*2.576
# 98% confidence is [-2.3sigma,+2.3sigma] so divide by 2*2.326
# 95% confidence is [-2sigma,+2sigma] so divide by 2*1.960
# 87% confidence is [-1.5*sigma,+1.5sigma] so divide by 2*1.514
# 68% confidence is [-sigma,+sigma] so divide by 2*0.9945


sigma_to_conv = {
    0.99: 2.576,
    0.98: 2.326,
    0.95: 1.960,
    0.87: 1.514,
    0.68: 0.9945,
}

def Confidence_numpy(hist, bins_mid, bin_width, confLevel = 0.87):
    ix = np.argmax(bins_mid>np.average(bins_mid, weights=hist))
    # print(ix, np.average(bins_mid, weights=hist))
    ixlow = ix
    ixhigh = ix
    nb = len(hist)
    ntot = np.sum(hist)
    nsum = hist[ix]
    width = bin_width
    # print("bin width", bin_width)
    if ntot==0: return 0
    while (nsum < confLevel * ntot):
        nlow = hist[ixlow-1] if ixlow>0 else 0
        nhigh = hist[ixhigh+1] if ixhigh<nb else 0
        if (nsum+max(nlow,nhigh) < confLevel * ntot):
            if (nlow>=nhigh and ixlow>0):
                nsum += nlow
                ixlow -=1
                width += bin_width
            elif ixhigh<nb:
                nsum += nhigh
                ixhigh+=1
                width += bin_width
            else: raise ValueError('BOOM')
        else:
            if (nlow>nhigh):
                width +=  bin_width * (confLevel * ntot - nsum) / nlow
            else:
                width +=  bin_width * (confLevel * ntot - nsum) / nhigh
            nsum = ntot
    # print(width)
    return width/(2* sigma_to_conv[confLevel])



# def Confidence(hist, confLevel = 0.87):
#     ix = hist.GetXaxis().FindBin(hist.GetMean())
#     ixlow = ix
#     ixhigh = ix
#     nb = hist.GetNbinsX()
#     ntot = hist.Integral()
#     nsum = hist.GetBinContent(ix)
#     width = hist.GetBinWidth(ix)
#     if ntot==0: return 0
#     while (nsum < confLevel * ntot):
#         nlow = hist.GetBinContent(ixlow-1) if ixlow>0 else 0
#         nhigh = hist.GetBinContent(ixhigh+1) if ixhigh<nb else 0
#         if (nsum+max(nlow,nhigh) < confLevel * ntot):
#             if (nlow>=nhigh and ixlow>0):
#                 nsum += nlow
#                 ixlow -=1
#                 width += hist.GetBinWidth(ixlow)
#             elif ixhigh<nb:
#                 nsum += nhigh
#                 ixhigh+=1
#                 width += hist.GetBinWidth(ixhigh)
#             else: raise ValueError('BOOM')
#         else:
#             if (nlow>nhigh):
#                 width += hist.GetBinWidth(ixlow-1) * (confLevel * ntot - nsum) / nlow
#             else:
#                 width += hist.GetBinWidth(ixhigh+1) * (confLevel * ntot - nsum) / nhigh
#             nsum = ntot
#     return width