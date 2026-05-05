import ROOT
import numpy as np
import pandas as pd
import helpers
import os
import pickle
import warnings
import argparse
import yaml
import matplotlib.pyplot as plt
import uproot
from hipe4ml import analysis_utils, plot_utils
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.model_handler import ModelHandler
from hipe4ml.analysis_utils import train_test_generator

# data to use
Real_Data = TreeHandler('AODDat.root', 'DF_2261906085717504/O2lambdatableml')
MC_Data = TreeHandler('AODMC.root', 'DF_2261906078815382/O2mclambdatableml')

# split data in Prompt, Non_Prompt and Background with cuts 
Prompt = MC_Data.get_subset(
    'fIsReco == 1 and fCosPA > 0.97 and fRadius > 3 and '
    '(fPDGCode == 3122 or fPDGCode == -3122) and '
    '(abs(fPDGCodeMother) != 3312 and abs(fPDGCodeMother) != 3322 and abs(fPDGCodeMother) != 3334)'
)

Non_Prompt = MC_Data.get_subset(
    'fIsReco == 1 and fCosPA > 0.97 and fRadius > 3 and '
    '(fPDGCode == 3122 or fPDGCode == -3122) and '
    '(abs(fPDGCodeMother) == 3312 or abs(fPDGCodeMother) == 3322 or abs(fPDGCodeMother) == 3334)')

#Background = Dati_MC.get_subset('fIsReco == 1 and fRadius > 3 and fCosPA > 0.97 and (fPDGCode != 3122 and fPDGCode != -3122)')
Background = Real_Data.get_subset('fRadius > 3 and fCosPA > 0.97 and (fMass < 1.098 or fMass > 1.145)', size = None )

# attributions of the labels and the samples of training and testing
train_test_data = train_test_generator([Background, Prompt, Non_Prompt],[0, 1, 2], test_size=0.5,random_state=42)
x_train, y_train, x_test, y_test = train_test_data

# downloading of the model trained
model_hdl = ModelHandler()
#model_hdlT.load_model_handler("Risultati_Bkg_in_DatiMC/model_Bkg_In_DatiMC.pkl")
model_hdl.load_model_handler("Risultati_Bkg_in_DatiReali/model_Bkg_In_DatiReali.pkl")


# creation of the function ndarray2th1 for the convertion from numpy to TH1 histograms
def ndarray2th1(arr, name, nbins, xmin, xmax):
    arr = np.asarray(arr).flatten()
    hist = ROOT.TH1F(name, name, nbins, xmin, xmax)
    for x in arr:
        if np.isfinite(x) and xmin <= x <= xmax:
            hist.Fill(float(x))

    return hist        

# definition of the real data prediction for prompt, nprompt, and background
x_rd = Real_Data.get_subset('fRadius > 3 and fCosPA > 0.97', size = 1000000) # rd = real data, x_rd = real data sample
y_pred_rd = model_hdl.predict(x_rd, output_margin=False) # y_pred_rd = real data predicted
x_rd_bdt = np.array([y[1] for y in y_pred_rd]) # real data score prompt

# definition of the test data predictions for prompt, non_prompt and background
y_pred_test = np.array(model_hdl.predict(x_test, output_margin=False)) # y_pred_t = test data predicted
y_pred_test_prompt_out = y_pred_test[y_test == 1, 1] # score prompt output for test data 
y_pred_test_nprompt_out = y_pred_test[y_test == 2, 1] # score non prompt output for test data
y_pred_test_bkg_out = y_pred_test[y_test == 0, 1] # score background output for test data


# creation of the values with bins
bdt_out = ROOT.RooRealVar("bdt_output", "BDT output", 0, 1)
Nbins_output_score = 100
bdt_out.setBins(Nbins_output_score)

# creation of the histogram of real data used for the fitting (h_rd = histo of real data) case of the prompt
h_rd = ndarray2th1(x_rd_bdt, "h_rd", Nbins_output_score, 0, 1)
bdt_rd = ROOT.RooDataHist("real_data", "real_data", ROOT.RooArgList(bdt_out), h_rd)

# creation of the Prompt, non_Prompt, and background histograms as elements used in the fitting

# case of prompt data test and his PDF template
h_prompt = ndarray2th1(y_pred_test_prompt_out, "h_prompt", Nbins_output_score, 0, 1)
dh_prompt = ROOT.RooDataHist("dh_prompt", "dh_prompt", ROOT.RooArgList(bdt_out), h_prompt)
pdf_prompt = ROOT.RooHistPdf("pdf_prompt", "pdf_prompt", ROOT.RooArgSet(bdt_out), dh_prompt)

# case of non_prompt data test and his PDF template                                         
h_nprompt = ndarray2th1(y_pred_test_nprompt_out, "h_nprompt", Nbins_output_score, 0, 1)
dh_nprompt = ROOT.RooDataHist("dh_nprompt", "dh_nprompt", ROOT.RooArgList(bdt_out), h_nprompt)
pdf_nprompt = ROOT.RooHistPdf("pdf_nprompt", "pdf_nprompt", ROOT.RooArgSet(bdt_out), dh_nprompt)

# case of background data test and his PDF template
h_bkg = ndarray2th1(y_pred_test_bkg_out, "h_bkg", Nbins_output_score, 0, 1)
dh_bkg = ROOT.RooDataHist("dh_bkg", "dh_bkg", ROOT.RooArgList(bdt_out), h_bkg)
pdf_bkg = ROOT.RooHistPdf("pdf_bkg", "pdf_bkg", ROOT.RooArgSet(bdt_out), dh_bkg)


print("Entries prompt template :", h_prompt.Integral())
print("Entries nprompt template:", h_nprompt.Integral())
print("Entries bkg template    :", h_bkg.Integral())
print("Entries data            :", h_rd.Integral())

# quantity values of prompt, non_prompt and background in 0 - 1e6
n_prompt = ROOT.RooRealVar("n_prompt", "n_prompt", 10000, 0, 1e6)
n_nprompt = ROOT.RooRealVar("n_nprompt", "n_nprompt", 10000, 0, 1e6)
n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", 10000, 0, 1e6)


# creation of the model
model = ROOT.RooAddPdf(
    "model", "",
    ROOT.RooArgList(pdf_prompt, pdf_nprompt, pdf_bkg),
    ROOT.RooArgList(n_prompt, n_nprompt, n_bkg))

# fit of the model
fit_result = model.fitTo(bdt_rd, ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True))
fit_result.Print()

# canvas
c = ROOT.TCanvas("c","",800,700)
c.SetLeftMargin(0.15)
c.SetBottomMargin(0.12)

# frame with title and axis
frame = bdt_out.frame(ROOT.RooFit.Title("BDT output fit"))
frame.GetXaxis().SetTitle("BDT score")
frame.GetYaxis().SetTitle("Entries")
frame.GetYaxis().SetTitleOffset(1.4)

# plotting real data
bdt_rd.plotOn(frame, ROOT.RooFit.Name("real data"), ROOT.RooFit.MarkerStyle(20),
                         ROOT.RooFit.MarkerSize(0.8))


# plotting total model
model.plotOn(frame, ROOT.RooFit.Name("model"), ROOT.RooFit.LineColor(ROOT.kBlack),
                 ROOT.RooFit.LineWidth(2))
# plotting components
model.plotOn(frame, ROOT.RooFit.Components("pdf_prompt"),
             ROOT.RooFit.LineColor(ROOT.kRed+1),
             ROOT.RooFit.LineStyle(ROOT.kDashed),
             ROOT.RooFit.LineWidth(2),
             ROOT.RooFit.Name("prompt"))

model.plotOn(frame, ROOT.RooFit.Components("pdf_nprompt"),
             ROOT.RooFit.LineColor(ROOT.kBlue+1),
             ROOT.RooFit.LineStyle(ROOT.kSolid),
             ROOT.RooFit.LineWidth(3),
             ROOT.RooFit.Name("nprompt"))

model.plotOn(frame, ROOT.RooFit.Components("pdf_bkg"),
             ROOT.RooFit.LineColor(ROOT.kGreen+2),
             ROOT.RooFit.LineStyle(ROOT.kDotted),
             ROOT.RooFit.LineWidth(3),
             ROOT.RooFit.Name("bkg"))

# legend
legend = ROOT.TLegend(0.6, 0.6, 0.88, 0.88)
legend.SetBorderSize(0)
legend.SetFillStyle(0)
legend.SetTextSize(0.035)

legend.AddEntry(frame.findObject("real data"), "real data", "lep")
legend.AddEntry(frame.findObject("model"), "total fit", "l")
legend.AddEntry(frame.findObject("prompt"), "prompt", "l")
legend.AddEntry(frame.findObject("nprompt"), "non_prompt", "l")
legend.AddEntry(frame.findObject("bkg"), "background", "l")


tot = n_prompt.getVal() + n_nprompt.getVal() + n_bkg.getVal()
print("Prompt     :", n_prompt.getVal())
print("Non prompt :", n_nprompt.getVal())
print("Background :", n_bkg.getVal())
print("Total      :", tot)

frame.Draw()
legend.Draw()

pave = ROOT.TPaveText(0.60, 0.65, 0.88, 0.88, "NDC")
pave.SetFillStyle(0)
pave.SetBorderSize(0)
pave.SetTextAlign(12)

pave.AddText(f"n_prompt = ({n_prompt.getVal():,.0f} ± {n_prompt.getError():,.0f})")
pave.AddText(f"n_nprompt = ({n_nprompt.getVal():,.0f} ± {n_nprompt.getError():,.0f})")
pave.AddText(f"n_bkg = ({n_bkg.getVal():,.0f} ± {n_bkg.getError():,.0f})")

pave.Draw()

# create and save root file
file_out = ROOT.TFile("fit_bdt.root", "RECREATE")
dir_bdt_output = file_out.mkdir("RealData - TestData")
dir_bdt_output.cd()
file_out.cd()

c.cd()
c.Modified()
c.Update()

c.Write("canvas_fit")
h_rd.Write()
h_prompt.Write()
h_nprompt.Write()
h_bkg.Write()
