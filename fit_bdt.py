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
Dati_Reali = TreeHandler('AODDat.root', 'DF_2261906085717504/O2lambdatableml')
Dati_MC = TreeHandler('AODMC.root', 'DF_2261906078815382/O2mclambdatableml')

# split data in Prompt, Non_Prompt and Background with cuts 
Prompt = Dati_MC.get_subset(
    'fIsReco == 1 and fCosPA > 0.97 and fRadius > 3 and '
    '(fPDGCode == 3122 or fPDGCode == -3122) and '
    '(abs(fPDGCodeMother) != 3312 and abs(fPDGCodeMother) != 3322 and abs(fPDGCodeMother) != 3334)'
)

Non_Prompt = Dati_MC.get_subset(
    'fIsReco == 1 and fCosPA > 0.97 and fRadius > 3 and '
    '(fPDGCode == 3122 or fPDGCode == -3122) and '
    '(abs(fPDGCodeMother) == 3312 or abs(fPDGCodeMother) == 3322 or abs(fPDGCodeMother) == 3334)')

#Background = Dati_MC.get_subset('fIsReco == 1 and fRadius > 3 and fCosPA > 0.97 and (fPDGCode != 3122 and fPDGCode != -3122)')
Background = Dati_Reali.get_subset('fRadius > 3 and fCosPA > 0.97 and (fMass < 1.098 or fMass > 1.145)', size = None )

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

# definition of the real data prediction for prompt
x_rd = Dati_Reali.get_subset('fRadius > 3 and fCosPA > 0.97', size = 1000000) # rd = real data, x_rd = real data sample
y_pred_rd = model_hdl.predict(x_rd, output_margin=False) # y_pred_rd = real data predicted
x_rd_bdt_prompt_out = np.array([y[1] for y in y_pred_rd]) # real data predicted sample

# definition of the test data predictions for prompt, non_prompt and background
y_pred_test = model_hdl.predict(x_test, output_margin=False) # y_pred_t = test data predicted
y_pred_test_prompt_out = np.array([y[1] for y in y_pred_test]) # prompt output for test data predicted
y_pred_test_nprompt_out = np.array([y[2] for y in y_pred_test]) # non prompt output for test data predicted
y_pred_test_bkg_out = np.array([y[0] for y in y_pred_test]) # background output for test data predicted


# creation of the values with bins
bdt_out = ROOT.RooRealVar("bdt_output", "BDT output", 0, 1)
Nbins_output_score = 100
bdt_out.setBins(Nbins_output_score)

# creation of the histogram of real data used for the fitting (h_rd = histo of real data) case of the prompt
h_rd = ndarray2th1(x_rd_bdt_prompt_out, "h_rd", Nbins_output_score, 0, 1)
bdt_prompt_rd = ROOT.RooDataHist("real_data", "real_data", ROOT.RooArgList(bdt_out), h_rd)

# creation of the Prompt, non_Prompt, and background histograms as elements used in the fitting

# case of prompt data test and his PDF template
h_prompt = ndarray2th1(y_pred_test_prompt_out, "h_prompt", Nbins_output_score, 0, 1)
dh_prompt = ROOT.RooDataHist("dh_prompt_test", "dh_prompt_test", ROOT.RooArgList(bdt_out), h_prompt)
pdf_prompt = ROOT.RooHistPdf("pdf_prompt", "pdf_prompt", ROOT.RooArgSet(bdt_out), dh_prompt)

# case of non_prompt data test and his PDF template                                         
h_nprompt = ndarray2th1(y_pred_test_nprompt_out, "h_nprompt", Nbins_output_score, 0, 1)
dh_nprompt = ROOT.RooDataHist("dh_nprompt_test", "dh_nprompt_test", ROOT.RooArgList(bdt_out), h_nprompt)
pdf_nprompt = ROOT.RooHistPdf("pdf_nprompt", "pdf_nprompt", ROOT.RooArgSet(bdt_out), dh_nprompt)

# case of background data test and his PDF template
h_bkg = ndarray2th1(y_pred_test_bkg_out, "h_bkg", Nbins_output_score, 0, 1)
dh_bkg = ROOT.RooDataHist("dh_bkg_test", "dh_bkg_test", ROOT.RooArgList(bdt_out), h_bkg)
pdf_bkg = ROOT.RooHistPdf("pdf_bkg", "pdf_bkg", ROOT.RooArgSet(bdt_out), dh_bkg)


# quantity values of prompt, non_prompt and background in 0 - 1e6
n_prompt = ROOT.RooRealVar("n_prompt", "n_prompt", 0, 1e6)
n_nprompt = ROOT.RooRealVar("n_nprompt", "n_nprompt", 0, 1e6)
n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", 0, 1e6)

# creation of the model
model = ROOT.RooAddPdf(
    "model", "model",
    ROOT.RooArgList(pdf_prompt, pdf_nprompt, pdf_bkg),
    ROOT.RooArgList(n_prompt, n_nprompt, n_bkg)
)

# fit of the model
model.fitTo(bdt_prompt_rd, ROOT.RooFit.Extended(True))
frame = bdt_out.frame()
bdt_prompt_rd.plotOn(frame)

# canvas
c = ROOT.TCanvas("c","",800,700)
c.SetLeftMargin(0.15)
c.SetBottomMargin(0.12)

# frame with title and axis
frame = bdt_out.frame(ROOT.RooFit.Title("BDT output fit"))
frame.GetXaxis().SetTitle("BDT score")
frame.GetYaxis().SetTitle("Entries")
frame.GetYaxis().SetTitleOffset(1.4)

# plot real data
bdt_prompt_rd.plotOn(frame, ROOT.RooFit.Name("real data"), ROOT.RooFit.MarkerStyle(20),
                         ROOT.RooFit.MarkerSize(0.8))
# plot total model
model.plotOn(frame, ROOT.RooFit.Name("model"), ROOT.RooFit.LineColor(ROOT.kBlack),
                 ROOT.RooFit.LineWidth(2))
# plot components
model.plotOn(frame, ROOT.RooFit.Components("pdf_prompt"),
             ROOT.RooFit.LineColor(ROOT.kRed+1),
             ROOT.RooFit.LineStyle(ROOT.kDashed),
             ROOT.RooFit.LineWidth(2),
             ROOT.RooFit.Name("prompt"))

model.plotOn(frame, ROOT.RooFit.Components("pdf_nprompt"),
             ROOT.RooFit.LineColor(ROOT.kBlue+1),
             ROOT.RooFit.LineStyle(ROOT.kDashed),
             ROOT.RooFit.LineWidth(2),
             ROOT.RooFit.Name("nprompt"))

model.plotOn(frame, ROOT.RooFit.Components("pdf_bkg"),
             ROOT.RooFit.LineColor(ROOT.kGreen+2),
             ROOT.RooFit.LineStyle(ROOT.kDashed),
             ROOT.RooFit.LineWidth(2),
             ROOT.RooFit.Name("bkg")) 
frame.Draw()

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
legend.Draw()

c.Update()

tot = n_prompt.getVal() + n_nprompt.getVal() + n_bkg.getVal()
print("Prompt     :", n_prompt.getVal())
print("Non prompt :", n_nprompt.getVal())
print("Background :", n_bkg.getVal())

# create and save root file
file_out = ROOT.TFile("fit_bdt.root", "RECREATE")
dir_bdt_output = file_out.mkdir("RealData - TestData")
dir_bdt_output.cd()

c.Write("canvas_fit")
frame.Write("roo_frame")
h_rd.Write()
h_prompt.Write()
h_nprompt.Write()
h_bkg.Write()

file_out.Close()

input ("press enter to exit")
