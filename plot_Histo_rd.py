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

# selection of real data
rd_sample = Dati_Reali.get_subset('fRadius > 3 and fCosPA > 0.97', size = 1000000) # rd(real data)

# =========================
# LOAD MODEL
# =========================
model_hdl = ModelHandler()
model_hdl.load_model_handler("Risultati_Bkg_in_DatiReali/model_Bkg_In_DatiReali.pkl")

# =========================
# DATA PREDICTION
# =========================
y_pred_rd = model_hdl.predict(rd_sample, output_margin=False) # rd(real data), prediction of real data
# =========================
# HISTOGRAMS
# =========================
nbins = 100
xmin, xmax = 0, 1
xmin_mass, xmax_mass = 1.08, 1.16

h_rd = {
    score: ROOT.TH1D(f"h_rd_s{score}", "", nbins, xmin, xmax)
    for score in [0, 1, 2]
}

h_fMass = { score: ROOT.TH1D(f"h_fMass_s{score}","", nbins, xmin_mass, xmax_mass) for score in[0,1,2]
    }    
df = rd_sample.get_data_frame()
# =========================
# FILL HISTOGRAMS
# =========================
cut = 0.8

for i in range(len(y_pred_rd)):
    mass = df.iloc[i]["fMass"]
    for score_index in [0,1,2]:
        score = y_pred_rd[i][score_index]
        h_rd[score_index].Fill(score)
        if y_pred_rd[i][score_index] > cut:
            h_fMass[score_index].Fill(mass)

            
        
# =========================
# NORMALIZATION
# =========================
def normalize(h):
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

for score in [0,1,2]:
    normalize(h_rd[score])
    normalize(h_fMass[score])

# =========================
# CANVAS AND LEGEND OF AXIS
# =========================
colors = {
    0: ROOT.kBlue,
    1: ROOT.kOrange,
    2: ROOT.kGreen}
    
for score_index in [0, 1, 2]:
    c = ROOT.TCanvas(f"c_{score_index}","", 800, 600)
    c.SetLogy()

    # titles of axis
    h_rd[score_index].GetXaxis().SetTitle(f"BDT score (class{score_index})")
    h_fMass[score_index].GetXaxis().SetTitle(f"Invariant Mass (GeV/c^{2})")
    h_rd[score_index].GetYaxis().SetTitle(f"normalized counts")
    h_fMass[score_index].GetYaxis().SetTitle(f"normalized counts")
    h_rd[score_index].SetLineColor(colors[score_index])
    h_fMass[score_index].SetLineColor(colors[score_index])
    h_rd[score_index].SetLineWidth(2)
    h_fMass[score_index].SetLineWidth(2)
    h_rd[score_index].SetTitle(f"Real data - class score {score_index}")
    h_fMass[score_index].SetTitle(f"Invariant Mass - class score {score_index}")
    h_rd[score_index].SetMinimum(1e-4)
    h_fMass[score_index].SetMinimum(1e-4)
    h_rd[score_index].Draw("HIST")
    h_fMass[score_index].Draw("HIST")
    c.Update()
    for score in [0, 1, 2]:
             h_rd[score].Sumw2(False)
             h_fMass[score].Sumw2(False)

# ==========================
# CREATE AND SAVE ROOT FILES
# ==========================
fout1 = ROOT.TFile("Histo_RealData_fMass.root", "RECREATE")
# directory of invariant Mass and saving
dir_fMass = fout1.mkdir("InvariantMass")
dir_fMass.cd()
for score in [0,1,2]:
    h_fMass[score].Write()

# directory and subdirectories of real data and saving
dir_real_data = fout1.mkdir("real data")
dir_real_data.cd()
dir_real_data_bkg = dir_real_data.mkdir("Background")
dir_real_data_Prompt = dir_real_data.mkdir("Prompt")
dir_real_data_NonPrompt = dir_real_data.mkdir("NonPrompt")

for score in [0,1,2]:
        if score == 0:
            dir_real_data_bkg.cd()
        elif score == 1:
            dir_real_data_Prompt.cd()
        else:
            dir_real_data_NonPrompt.cd()
        h_rd[score].Write()
fout1.Close()
