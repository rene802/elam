###################################################################################################################################
# Script for RooFit and Pull plot with separate canvas, to have a better visualization of the fit results and the pull distribution
####################################################################################################################################
import json
import math
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import copy
import os

os.chdir("/Users/estheradasiakam/Tesi")
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

import shap
import subprocess
import ROOT

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.optimize import curve_fit

import xgboost as xgb

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml import plot_utils

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree

##################
# Loading data
##################
Real_Data = TreeHandler('AODDat.root', 'DF_2261906085717504/O2lambdatableml')
MC_Data = TreeHandler('AODMC.root', 'DF_2261906078815382/O2mclambdatableml')

###################
# Data selection
###################
Prompt = MC_Data.get_subset('fIsReco == 1 and ''fCosPA > 0.97 and ''fRadius > 3 and ''(fPDGCode == 3122 or fPDGCode == -3122) '
'and ''(abs(fPDGCodeMother) != 3312 and ''abs(fPDGCodeMother) != 3322 and ' 'abs(fPDGCodeMother) != 3334)')

Non_Prompt = MC_Data.get_subset('fIsReco == 1 and ''fCosPA > 0.97 and ''fRadius > 3 and ' '(fPDGCode == 3122 or fPDGCode == -3122) and '
    '(abs(fPDGCodeMother) == 3312 or ''abs(fPDGCodeMother) == 3322 or ''abs(fPDGCodeMother) == 3334)')

Background_mc = MC_Data.get_subset('fIsReco == 1 and ''fCosPA > 0.97 and ''fRadius > 3 and ''(fPDGCode != 3122 or fPDGCode != -3122)',size=None) # mc = monte carlo
Background_rd = Real_Data.get_subset('fRadius > 3 and ''fCosPA > 0.97 and ''(fMass < 1.098 or fMass > 1.145)',size=None) # rd = real data

#################################################
# Split of train and test data sets for real data
#################################################
train_test_data_rd = train_test_generator([Background_rd, Prompt, Non_Prompt],[0, 1, 2],test_size=0.5,random_state=42)
x_train_rd, y_train_rd, x_test_rd, y_test_rd = train_test_data_rd

##############################################################
# Loading of the model trained on the background in real data
##############################################################
model_hdl = ModelHandler()
model_hdl.load_model_handler("Risultati_Bkg_in_DatiReali/model_Bkg_In_DatiReali.pkl")

############################################
# Creation of parquet file for real data
#############################################
parquet_file = "final_output.parquet.gzip"

##########################################################################################
# Check if the parquet file already exists, if not create it by applying the model to the
#  real data and saving the output in parquet format, to have a faster loading of the 
# data for the fitting and the pull plot.
##########################################################################################
if not os.path.exists(parquet_file):
    print("\nCreating parquet file...\n")
    handler = Real_Data
    handler.apply_model_handler(model_hdl, output_margin=False)
    handler.write_df_to_parquet_files("final_output")
    print("\nParquet created.\n")

#############################################################
# Loadinf of the parquet file and reading in pandas dataframe
#############################################################
print("\nLoading parquet...\n")
df_parquet = pd.read_parquet(parquet_file)

##################################################################################
# Function of convertion from numpy array to TH1 histogram, and function for
#  the normalization of the histograms, to have a better visualization
##################################################################################
def ndarray2th1(arr, name, nbins, xmin, xmax):
    hist = ROOT.TH1F(name, name, nbins, xmin, xmax)
    hist.SetDirectory(0)
    for x in np.asarray(arr).flatten():
        if np.isfinite(x) and xmin <= x <= xmax:
            hist.Fill(float(x))
    return hist

def normalize(h):
    n = h.Integral()
    if n > 0:
        h.Scale(1.0 / n)
    for i in range(1, h.GetNbinsX() + 1):
        if h.GetBinContent(i) == 0:
            h.SetBinContent(i, 1e-6)

##############################
# Preparation of the real data
##############################
rd_df = df_parquet.sample(n=40000, random_state=42)
x_rd_bdt = rd_df["model_output_1"].to_numpy()

##############################################################
# Preparation of the test data, with the application of the 
# model to the test data and the creation of a dataframe with
# the output of the model.
###############################################################
#test_df = pd.DataFrame(x_test_rd.copy())
#test_df["label"] = y_test_rd
#test_hdl = TreeHandler()
#test_hdl.set_data_frame(test_df)
#test_hdl.apply_model_handler(model_hdl, output_margin=False)
#df_test = test_hdl.get_data_frame()

test_parquet = "test_templates.parquet.gzip"
if not os.path.exists(test_parquet):
    print("\nCreating test parquet file...\n")
    df_test = pd.DataFrame(x_test_rd.copy())
    df_test["label"] = y_test_rd
    df_test = df_test.sample(n=40000, random_state=42)
    test_hdl = TreeHandler()
    test_hdl.set_data_frame(df_test)
    test_hdl.apply_model_handler(model_hdl, output_margin=False)
    test_hdl.write_df_to_parquet_files("test_templates")
    print("\nTest parquet created.\n")
else:
    print("\nLoading test parquet...\n")
    test_df = pd.read_parquet(test_parquet)

############################################################################
# Separation of the test data in prompt, non_prompt and background with the
# selection of the output score of the model for each category
############################################################################
y_pred_test_prompt_out = (df_test[df_test["label"] == 1]["model_output_1"].to_numpy())
y_pred_test_nprompt_out = (df_test[df_test["label"] == 2]["model_output_1"].to_numpy())
y_pred_test_bkg_out = (df_test[df_test["label"] == 0]["model_output_1"].to_numpy())

############################################################################
# Creation of the variable to fit, with the definition of the number of bins
############################################################################
bdt_out = ROOT.RooRealVar("bdt_output", "BDT output", 0, 1)
Nbins = 40
bdt_out.setBins(Nbins)

###############################
# Creation of the histograms 
###############################
h_rd = ndarray2th1(x_rd_bdt, "h_rd", Nbins, 0, 1)

h_prompt = ndarray2th1(y_pred_test_prompt_out, "h_prompt", Nbins, 0, 1)
h_nprompt = ndarray2th1(y_pred_test_nprompt_out, "h_nprompt", Nbins, 0, 1)
h_bkg = ndarray2th1(y_pred_test_bkg_out, "h_bkg", Nbins, 0, 1)

##################################
# Normalization of the histograms
##################################
for h in [h_rd, h_prompt, h_nprompt, h_bkg]:
    h.Smooth(1)
    normalize(h)

#################################################################
# Creation of the RooDataHist for the real data and the templates
#################################################################
bdt_rd = ROOT.RooDataHist("real_data", "real_data", ROOT.RooArgList(bdt_out), h_rd)

dh_prompt = ROOT.RooDataHist("dh_prompt", "dh_prompt", ROOT.RooArgList(bdt_out), h_prompt)
dh_nprompt = ROOT.RooDataHist("dh_nprompt", "dh_nprompt", ROOT.RooArgList(bdt_out), h_nprompt)
dh_bkg = ROOT.RooDataHist("dh_bkg", "dh_bkg", ROOT.RooArgList(bdt_out), h_bkg)

##########################################################################
# Creation of the PDFs for the prompt, non_prompt and background templates
##########################################################################
pdf_prompt = ROOT.RooHistPdf("pdf_prompt", "pdf_prompt", ROOT.RooArgSet(bdt_out), dh_prompt)
pdf_nprompt = ROOT.RooHistPdf("pdf_nprompt", "pdf_nprompt", ROOT.RooArgSet(bdt_out), dh_nprompt)
pdf_bkg = ROOT.RooHistPdf("pdf_bkg", "pdf_bkg", ROOT.RooArgSet(bdt_out), dh_bkg)

##################################
# Fitting of the model to the data
##################################
n_tot = len(x_rd_bdt)
n_prompt = ROOT.RooRealVar("n_prompt", "n_prompt", 0.6 * n_tot, 0, n_tot)
n_nprompt = ROOT.RooRealVar("n_nprompt", "n_nprompt", 0.3 * n_tot, 0, n_tot)
n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", 0.1 * n_tot, 0, n_tot)

model = ROOT.RooAddPdf("model", "", ROOT.RooArgList(pdf_prompt, pdf_nprompt, pdf_bkg), ROOT.RooArgList(n_prompt, n_nprompt, n_bkg))
fit_result = model.fitTo(bdt_rd, ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True))
fit_result.Print()

##########################
# Main frame for plotting
###########################
frame = bdt_out.frame(ROOT.RooFit.Title("BDT output fit"))
frame.GetYaxis().SetTitle("Entries")
frame.GetYaxis().SetTitleSize(0.05)
frame.GetYaxis().SetTitleOffset(1.2)
frame.GetXaxis().SetLabelSize(0.04)

###########################
# Plotting of the real data
###########################
bdt_rd.plotOn(frame, ROOT.RooFit.Name("data"), ROOT.RooFit.MarkerStyle(20), ROOT.RooFit.MarkerSize(0.8))

################
# Model plotting
################
model.plotOn(frame, ROOT.RooFit.Name("model"), ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.LineWidth(2))

#####################
# Components plotting
#####################
model.plotOn(frame,ROOT.RooFit.Components("pdf_prompt"),ROOT.RooFit.LineColor(ROOT.kRed + 1),ROOT.RooFit.LineStyle(ROOT.kDashed),ROOT.RooFit.LineWidth(2),ROOT.RooFit.Name("prompt"))
model.plotOn(frame,ROOT.RooFit.Components("pdf_nprompt"),ROOT.RooFit.LineColor(ROOT.kBlue + 1),ROOT.RooFit.LineWidth(3),ROOT.RooFit.Name("nprompt"))
model.plotOn(frame,ROOT.RooFit.Components("pdf_bkg"),ROOT.RooFit.LineColor(ROOT.kGreen + 2),ROOT.RooFit.LineStyle(ROOT.kDotted),ROOT.RooFit.LineWidth(3),ROOT.RooFit.Name("bkg"))

##################
# Chi2 calculation
##################
npar = fit_result.floatParsFinal().getSize()
chi2 = frame.chiSquare("model","data", npar)

############
# Legend
############
legend = ROOT.TLegend(0.60, 0.60, 0.88, 0.88)
legend.SetBorderSize(0)
legend.SetFillStyle(0)
legend.SetTextSize(0.035)
legend.AddEntry(frame.findObject("data"), "real data", "lep")
legend.AddEntry(frame.findObject("model"), "total fit", "l")
legend.AddEntry(frame.findObject("prompt"), "prompt", "l")
legend.AddEntry(frame.findObject("nprompt"), "non_prompt", "l")
legend.AddEntry(frame.findObject("bkg"), "background", "l")

############################
# Pave text with fit results
############################
pave = ROOT.TPaveText(0.60, 0.45, 0.88, 0.58, "NDC")
pave.SetFillStyle(0)
pave.SetBorderSize(0)
pave.SetTextAlign(12)
pave.AddText(f"#chi^{{2}}\n  = \n{chi2:.2f}")
pave.AddText(f"n_prompt = {n_prompt.getVal():,.2f} #pm {n_prompt.getError():,.2f}")
pave.AddText(f"n_nprompt = {n_nprompt.getVal():,.2f} #pm {n_nprompt.getError():,.2f}")
pave.AddText(f"n_bkg = {n_bkg.getVal():,.2f} #pm {n_bkg.getError():,.2f}")

###################
# Fit canvas
####################
#ROOT.gROOT.SetBatch(True)
#c_fit = ROOT.TCanvas("c_fit","", 900, 700)
#frame.Draw()
#legend.Draw()
#pave.Draw()
#c_fit.SaveAs("fit_only.png")

##################
# Pull histogram
###################
pullHist = frame.pullHist("data", "model")

###################
# Pull frame
###################
pullFrame = bdt_out.frame(ROOT.RooFit.Title("Pull distribution"))
pullFrame.addPlotable(pullHist, "P")
pullFrame.GetYaxis().SetTitle("n#sigma")
pullFrame.GetYaxis().SetNdivisions(505)
pullFrame.GetYaxis().SetTitleSize(0.10)
pullFrame.GetYaxis().SetTitleOffset(0.4)
pullFrame.GetYaxis().SetLabelSize(0.08)
pullFrame.GetXaxis().SetTitle("BDT score")
pullFrame.GetXaxis().SetTitleSize(0.12)
pullFrame.GetXaxis().SetLabelSize(0.10)
pullFrame.SetMinimum(-5)
pullFrame.SetMaximum(5)


###################
# Fit canvas
####################
ROOT.gROOT.SetBatch(True)
c_fit = ROOT.TCanvas("c_fit","", 900, 900)
#frame.Draw()
#legend.Draw()
#pave.Draw()
#c_fit.SaveAs("fit_only.png")

################
# Upper pad for fit
################
pad1 = ROOT.TPad("pad1", "pad1", 0, 0.30, 1, 1)
pad1.SetBottomMargin(0.02)

####################
# Lower pad for pull
####################
pad2 = ROOT.TPad("pad2", "pad2", 0, 0, 1, 0.30)
pad2.SetTopMargin(0.05)
pad2.SetBottomMargin(0.30)
pad1.Draw()
pad2.Draw()

#######################
# Draw fit in upper pad
########################
pad1.cd()
frame.Draw()
legend.Draw()
pave.Draw()

#######################
# Draw pull in lower pad
#########################
pad2.cd()
pullFrame.Draw()
line = ROOT.TLine(0, 0, 1, 0)
line.SetLineColor(ROOT.kRed)
line.SetLineStyle(2)
line.Draw("same")

##########################
# Save the combined canvas
##########################
c_fit.SaveAs("fit_with_pull.png")

#################
# Pull canvas
#################
#c_pull = ROOT.TCanvas("c_pull","", 900, 500)
#pullFrame.Draw()
#line = ROOT.TLine(0, 0, 1, 0)
#line.SetLineColor(ROOT.kRed)
#line.SetLineStyle(2)
#line.Draw("same")
#c_pull.SaveAs("pull_only.png")

##############################################
# Save ROOT file with histograms and canvases
##############################################
outfile = ROOT.TFile("fit_bdt_prompt_pull.root", "RECREATE")
h_rd.Write()
h_prompt.Write()
h_nprompt.Write()
h_bkg.Write()
c_fit.Write()
c_fit.Write()
outfile.Close()

#################
# Print results
#################
tot = (n_prompt.getVal() + n_nprompt.getVal() + n_bkg.getVal())
print(f"\nχ² = {chi2:.2f}\n")
print("Prompt     :", n_prompt.getVal())
print("Non prompt :", n_nprompt.getVal())
print("Background :", n_bkg.getVal())
print("Total      :", tot)
