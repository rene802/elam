# Importations of libraries
import ROOT
import numpy as np
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.model_handler import ModelHandler
from hipe4ml.analysis_utils import train_test_generator

# =========================
# STYLE 
# =========================
ROOT.gROOT.SetBatch(False)
ROOT.gStyle.SetOptStat(1110)
ROOT.gStyle.SetTitleSize(0.045, "XY")
ROOT.gStyle.SetLabelSize(0.04, "XY")
ROOT.gStyle.SetPadLeftMargin(0.12)
ROOT.gStyle.SetPadBottomMargin(0.12)
ROOT.gStyle.SetPadTopMargin(0.05)
ROOT.gStyle.SetPadRightMargin(0.05)

# =========================
# LOAD DATA
# =========================
Dati_MC = TreeHandler('AODMC.root', 'DF_2261906078815382/O2mclambdatableml')
Dati_Reali = TreeHandler('AODDat.root', 'DF_2261906085717504/O2lambdatableml')

Prompt = Dati_MC.get_subset(
    'fIsReco == 1 and fCosPA > 0.97 and fRadius > 3 and '
    '(fPDGCode == 3122 or fPDGCode == -3122) and '
    '(abs(fPDGCodeMother) != 3312 and abs(fPDGCodeMother) != 3322 and abs(fPDGCodeMother) != 3334)'
)

Non_Prompt = Dati_MC.get_subset(
    'fIsReco == 1 and fCosPA > 0.97 and fRadius > 3 and '
    '(fPDGCode == 3122 or fPDGCode == -3122) and '
    '(abs(fPDGCodeMother) == 3312 or abs(fPDGCodeMother) == 3322 or abs(fPDGCodeMother) == 3334)'
)

#Background = Dati_MC.get_subset('fIsReco == 1 and fRadius > 3 and fCosPA > 0.97 and (fPDGCode != 3122 and fPDGCode != -3122)')
Background = Dati_Reali.get_subset('fRadius > 3 and fCosPA > 0.97 and (fMass < 1.098 or fMass > 1.145)', size = None )

# =========================
# TRAIN / TEST
# =========================
train_test_data = train_test_generator(
    [Background, Prompt, Non_Prompt],
    [0, 1, 2],
    test_size=0.5,
    random_state=42
)

# =========================
# LOAD MODEL
# =========================
model_hdlT = ModelHandler()
#model_hdlT.load_model_handler("Risultati_Bkg_in_DatiMC/model_Bkg_In_DatiMC.pkl")
model_hdlT.load_model_handler("Risultati_Bkg_in_DatiReali/model_Bkg_In_DatiReali.pkl")

# =========================
# DATA
# =========================
x_train, y_train, x_test, y_test = train_test_data

y_pred_train = model_hdlT.predict(x_train)
y_pred_test = model_hdlT.predict(x_test)

# =========================
# HISTOGRAMS
# =========================
nbins = 100
xmin, xmax = 0, 1

h_train = {
      score: {
          label: ROOT.TH1D(f"h_train_s{score}_c{label}", "", nbins, xmin, xmax) for label in [0, 1, 2]
          }
          for score in [0, 1, 2]
}

h_test = {
      score: {
          label: ROOT.TH1D(f"h_test_s{score}_c{label}", "", nbins, xmin, xmax)
for label in [0, 1, 2]
          }
          for score in [0, 1, 2]
}

# =========================
# FILL HISTOGRAMS
# =========================

# filling for training, with prediction in Background, Prompt, NonPrompt
for i in range(len(y_train)):
    label = y_train[i]
    for score_index in [0, 1, 2]:
        score = y_pred_train[i][score_index]
        h_train[score_index][label].Fill(score)        

 # filling for test, with prediction in Background, Prompt, NonPrompt       
for i in range(len(y_test)):
    label = y_test[i]
    for score_index in [0, 1, 2]:
        score = y_pred_test[i][score_index]
        h_test[score_index][label].Fill(score)

# =========================
# NORMALIZATION
# =========================
def normalize(h):
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

for score in [0,1,2]:
    for label in [0,1,2]:
        normalize(h_train[score][label])
        normalize(h_test[score][label])

# =========================
# STYLE FOR TRAIN AND TEST
# =========================
colors = {
    0: ROOT.kBlue,
    1: ROOT.kOrange,
    2: ROOT.kGreen
}
for score in [0,1,2]:
    for label in [0,1,2]:
        # train filled
        h_train[score][label].SetFillColorAlpha(colors[label], 0.35)
        h_train[score][label].SetLineColor(colors[label])

for score in [0,1,2]:
    for label in [0,1,2]:
        # test markers
        h_test[score][label].SetMarkerStyle(20)
        h_test[score][label].SetMarkerSize(0.8)
        h_test[score][label].SetMarkerColor(colors[label])
        h_test[score][label].SetLineColor(colors[label])

labels = {
    0: "Background",
    1: "Prompt",
    2: "NonPrompt"}        


# ================================================================
# CANVAS AND LEGEND FOR THE 3 TYPES, BACKGROUND, PROMPT, NONPROMPT
# ================================================================
for score_index in [0, 1, 2]:
    c = ROOT.TCanvas(f"c_{score_index}","", 800, 600)
    c.SetLogy()

    # titles of axis
    for label in [0, 1, 2]:
        h_test[score_index][label].GetXaxis().SetTitle(f"BDT score (class{score_index})")
        h_test[score_index][label].GetYaxis().SetTitle(f"normalized counts")
        h_train[score_index][label].GetXaxis().SetTitle(f"BDT score (class{score_index})")
        h_train[score_index][label].GetYaxis().SetTitle(f"normalized counts")

    # range for log scale
    h_train[score_index][0].SetMinimum(1e-4)
    h_train[score_index][0].SetMaximum(10)

    # draw train
    h_train[score_index][0].Draw("HIST")
    h_train[score_index][1].Draw("HIST SAME")
    h_train[score_index][2].Draw("HIST SAME")

    # draw test
    h_test[score_index][0].Draw("E SAME")
    h_test[score_index][1].Draw("E SAME")
    h_test[score_index][2].Draw("E SAME")
    

    leg = ROOT.TLegend(0.55, 0.6, 0.88, 0.88)
    for label in [0,1,2]:
        leg.AddEntry(h_train[score_index][label], f"{labels[label]} (train)", "f")
        leg.AddEntry(h_test[score_index][label], f"{labels[label]} (test)", "l")

    leg.Draw()
    c.Update()
    
    for score in [0, 1, 2]:
        for label in [0, 1, 2]:
             h_test[score][label].Sumw2(False)
             h_train[score][label].Sumw2(False)
# ==========================
# CREATE AND SAVE ROOT FILES
# ==========================
fout = ROOT.TFile("all_Histo_bkg_in_DatiReali.root", "RECREATE")
# main directories
dir_train = fout.mkdir("train")
dir_test = fout.mkdir("test")

# subdirectories
# for training
dir_train.cd() 
dir_train_bkg = dir_train.mkdir("Background")
dir_train_Prompt = dir_train.mkdir("Prompt")
dir_train_NonPrompt = dir_train.mkdir("NonPrompt")

# for test
dir_test.cd() 
dir_test_bkg = dir_test.mkdir("Background")
dir_test_Prompt = dir_test.mkdir("Prompt")
dir_test_NonPrompt = dir_test.mkdir("NonPrompt")

# save the histograms in the good files

for score in [0,1,2]:
    for label in [0,1,2]:
        # training
        if label == 0:
            dir_train_bkg.cd()
        elif label == 1:
            dir_train_Prompt.cd()
        else:
            dir_train_NonPrompt.cd()

        h_train[score][label].Write()

        # for test
        if label == 0:
            dir_test_bkg.cd()
        elif label == 1:
            dir_test_Prompt.cd()
        else:
            dir_test_NonPrompt.cd()

        h_test[score][label].Write()
fout.Close()
