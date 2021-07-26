import ROOT
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from array import array

f = ROOT.TFile("ntuple_SSWW_SM.root")
h = f.Get("SSWW_SM_nums")
xsec = h.GetBinContent(1)
sumw = h.GetBinContent(2)
normalization = xsec * 1000. * 100 / (sumw)
normalization_2 = xsec * 1000. * 350 / (sumw)

sm = np.loadtxt("plots_relu/loss_sm_test.csv")

dfAll = ROOT.RDataFrame("SSWW_SM","ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
SM = pd.DataFrame.from_dict(df.AsNumpy(["ptj1", "w"]))
weights_SM = SM["w"].to_numpy()
wx_train, wx_test, wy_train, wy_test = train_test_split(weights_SM, weights_SM, test_size=0.2, random_state=1)

h = ROOT.TH1D("h_sm", "h_sm", 100, 0., 0.02)
h.SetLineColor(ROOT.kRed)
h.FillN(len(sm), array('d', sm), array('d', wx_test) )
h.Scale(normalization*5)

h2 = ROOT.TH1D("h_sm_2", "h_sm_2", 100, 0., 0.02)
h2.SetLineColor(ROOT.kBlue)
h2.FillN(len(sm), array('d', sm), array('d', wx_test) )
h2.Scale(normalization_2*5)

ROOT.gStyle.SetOptStat(0)
leg = ROOT.TLegend(0.89, 0.89, 0.7, 0.7)
leg.SetBorderSize(0)
leg.AddEntry(h, "100fb^{-1}", "F")
leg.AddEntry(h2, "350fb^{-1}", "F")
c  = ROOT.TCanvas("c", "c", 1000, 1000)

h2.Draw("hist")
h.Draw("hist same")

leg.Draw()
c.Draw()

c.SaveAs("prova.png")
print(h.Integral())
