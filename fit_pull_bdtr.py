import pandas as pd
import ROOT
import numpy as np
import os
from config import model_dir, plot_dir, root_dir, log_dir

# configuration
parquet_file = os.path.join(model_dir,"p_output.parquet.gzip")
test_parquet = os.path.join(model_dir,"testp_output.parquet.gzip")
output_root = os.path.join(root_dir,"fit_bdt_prompt_pull.root")

# utility functions
def ndarray2th1(arr, name, nbins, xmin, xmax):
    if xmin == xmax:
        xmax += 1e-6
    hist = ROOT.TH1F(name, name, nbins, xmin, xmax)
    hist.Sumw2()
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

# real data reading parquet file in pandas dataframe
print("\nLoading parquet...\n")
df_parquet = pd.read_parquet(parquet_file)
df_parquet = df_parquet[(df_parquet['fMass'] > 1.113) & (df_parquet['fMass'] < 1.118)].copy()
rd_df = df_parquet.copy()

# test data reading parquet file in pandas dataframe
df_test = pd.read_parquet(test_parquet)
df_test = df_test.copy()

# extract bdt output
x_rd_bdt = rd_df["model_output_1"].to_numpy()
y_pred_test_prompt_out = (df_test[df_test["label"] == 1]["model_output_1"].to_numpy())
y_pred_test_nprompt_out = (df_test[df_test["label"] == 2]["model_output_1"].to_numpy())
y_pred_test_bkg_out = (df_test[df_test["label"] == 0]["model_output_1"].to_numpy())

# creation of the variables to fit
bdt_out = ROOT.RooRealVar("bdt_output", "BDT output", 0, 1)
Nbins = 60
bdt_out.setBins(Nbins)

print("Prompt entries :", len(y_pred_test_prompt_out))
print("Non prompt entries :", len(y_pred_test_nprompt_out))
print("Background entries :", len(y_pred_test_bkg_out))

if len(y_pred_test_prompt_out) == 0:
    raise ValueError("Prompt template empty")
if len(y_pred_test_nprompt_out) == 0:
    raise ValueError("Non-prompt template empty")
if len(y_pred_test_bkg_out) == 0:
    raise ValueError("Background template empty")

# creation of root histograms
h_rd = ndarray2th1(x_rd_bdt, "h_rd", Nbins, 0, 1)

h_prompt = ndarray2th1(y_pred_test_prompt_out, "h_prompt", Nbins, 0, 1)
h_nprompt = ndarray2th1(y_pred_test_nprompt_out, "h_nprompt", Nbins, 0, 1)
h_bkg = ndarray2th1(y_pred_test_bkg_out, "h_bkg", Nbins, 0, 1)

# normalization of histograms
for h in [h_prompt, h_nprompt, h_bkg]:
    #h.Smooth(1)
    normalize(h)

# creation of the RooDataHist for the real data and the templates
bdt_rd = ROOT.RooDataHist("real_data", "real_data", ROOT.RooArgList(bdt_out), h_rd)

dh_prompt = ROOT.RooDataHist("dh_prompt", "dh_prompt", ROOT.RooArgList(bdt_out), h_prompt)
dh_nprompt = ROOT.RooDataHist("dh_nprompt", "dh_nprompt", ROOT.RooArgList(bdt_out), h_nprompt)
dh_bkg = ROOT.RooDataHist("dh_bkg", "dh_bkg", ROOT.RooArgList(bdt_out), h_bkg)

# creation of the PDFs for the prompt, non_prompt and background templates
pdf_prompt = ROOT.RooHistPdf("pdf_prompt", "pdf_prompt", ROOT.RooArgSet(bdt_out), dh_prompt)
pdf_nprompt = ROOT.RooHistPdf("pdf_nprompt", "pdf_nprompt", ROOT.RooArgSet(bdt_out), dh_nprompt)
pdf_bkg = ROOT.RooHistPdf("pdf_bkg", "pdf_bkg", ROOT.RooArgSet(bdt_out), dh_bkg)

# Fitting of the model to the data
n_tot = len(x_rd_bdt)
n_prompt = ROOT.RooRealVar("n_prompt", "n_prompt", 0.7 * n_tot, 0, n_tot)
n_nprompt = ROOT.RooRealVar("n_nprompt", "n_nprompt", 0.3 * n_tot, 0, n_tot)
n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", 0.1 * n_tot, 0, n_tot)

model = ROOT.RooAddPdf("model", "", ROOT.RooArgList(pdf_prompt, pdf_nprompt, pdf_bkg), ROOT.RooArgList(n_prompt, n_nprompt, n_bkg))
fit_result = model.fitTo(bdt_rd, ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True))
fit_result.Print()

# main frame for plotting
frame = bdt_out.frame(ROOT.RooFit.Title("BDT output fit"))
frame.GetYaxis().SetTitle("Entries")
frame.GetYaxis().SetTitleSize(0.05)
frame.GetYaxis().SetTitleOffset(1.2)
frame.GetXaxis().SetLabelSize(0.04)

# Plotting of the real data
bdt_rd.plotOn(frame, ROOT.RooFit.Name("data"), ROOT.RooFit.MarkerStyle(20), ROOT.RooFit.MarkerSize(0.8))
model.plotOn(frame, ROOT.RooFit.Name("model"), ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.LineWidth(2))

# Components plotting
model.plotOn(frame,ROOT.RooFit.Components("pdf_prompt"),ROOT.RooFit.LineColor(ROOT.kOrange + 2),ROOT.RooFit.LineStyle(ROOT.kDashed),ROOT.RooFit.LineWidth(2),ROOT.RooFit.Name("prompt"))
model.plotOn(frame,ROOT.RooFit.Components("pdf_nprompt"),ROOT.RooFit.LineColor(ROOT.kGreen + 2),ROOT.RooFit.LineWidth(3),ROOT.RooFit.Name("nprompt"))
model.plotOn(frame,ROOT.RooFit.Components("pdf_bkg"),ROOT.RooFit.LineColor(ROOT.kBlue + 2),ROOT.RooFit.LineStyle(ROOT.kDotted),ROOT.RooFit.LineWidth(3),ROOT.RooFit.Name("bkg"))

# Chi2 calculation
npar = fit_result.floatParsFinal().getSize()
chi2 = frame.chiSquare("model","data", npar)

#  Legend
legend = ROOT.TLegend(0.60, 0.60, 0.88, 0.88)
legend.SetBorderSize(0)
legend.SetFillStyle(0)
legend.SetTextSize(0.035)
legend.AddEntry(frame.findObject("data"), "real data", "lep")
legend.AddEntry(frame.findObject("model"), "total fit", "l")
legend.AddEntry(frame.findObject("prompt"), "prompt", "l")
legend.AddEntry(frame.findObject("nprompt"), "non_prompt", "l")
legend.AddEntry(frame.findObject("bkg"), "background", "l")

#  Pave text with fit results
pave = ROOT.TPaveText(0.60, 0.45, 0.88, 0.58, "NDC")
pave.SetFillStyle(0)
pave.SetBorderSize(0)
pave.SetTextAlign(12)
pave.AddText(f"#chi^{{2}}\n  = \n{chi2:.2f}")
pave.AddText(f"n_prompt = {n_prompt.getVal():.2f} #pm {n_prompt.getError():.2f}")
pave.AddText(f"n_nprompt = {n_nprompt.getVal():.2f} #pm {n_nprompt.getError():.2f}")
pave.AddText(f"n_bkg = {n_bkg.getVal():.2f} #pm {n_bkg.getError():.2f}")

# Pull histogram
pullHist = frame.pullHist("data", "model")

# Distribution of pull values
h_pull_dist = ROOT.TH1F("h_pull_dist","Pull distribution;Pull;Entries",100,-100,100)

# Fill pull distribution from pull histogram bins
for i in range(pullHist.GetN()):
    pull_value = pullHist.GetY()[i]
    if np.isfinite(pull_value):
        h_pull_dist.Fill(pull_value)

# Gaussian fit of pull distribution        
gaus_fit = ROOT.TF1("gaus_fit", "gaus", -100, 100)
h_pull_dist.Fit(gaus_fit, "RQ")

# Canvas for pull distribution
c_pull_dist = ROOT.TCanvas("c_pull_dist","Pull distribution",800,600)

# Style
h_pull_dist.SetTitle("Pull distribution")
h_pull_dist.GetXaxis().SetTitle("Pull")
h_pull_dist.GetYaxis().SetTitle("Entries")
h_pull_dist.SetLineColor(ROOT.kBlue + 1)
h_pull_dist.SetLineWidth(2)
h_pull_dist.Draw("HIST")

# save image
c_pull_dist.SaveAs(os.path.join(plot_dir,"pull_distribution.png"))

# Pull frame
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
pullFrame.SetMinimum(-100)
pullFrame.SetMaximum(100)

# Fit canvas
ROOT.gROOT.SetBatch(True)
c_fit = ROOT.TCanvas("c_fit","", 900, 900)

# Upper pad for fit
pad1 = ROOT.TPad("pad1", "pad1", 0, 0.30, 1, 1)
pad1.SetBottomMargin(0.02)

# Lower pad for pull
pad2 = ROOT.TPad("pad2", "pad2", 0, 0, 1, 0.30)
pad2.SetTopMargin(0.05)
pad2.SetBottomMargin(0.30)
pad1.Draw()
pad2.Draw()

# Draw fit in upper pad
pad1.cd()
frame.Draw()
legend.Draw()
pave.Draw()

# Draw pull in lower pad
pad2.cd()
pullFrame.Draw()
line = ROOT.TLine(0, 0, 1, 0)
line.SetLineColor(ROOT.kRed)
line.SetLineStyle(2)
line.Draw("same")

# Save the combined canvas
c_fit.SaveAs(os.path.join(plot_dir,"fit_with_pull.png"))

# Save ROOT file with histograms and canvases
outfile = ROOT.TFile(output_root, "RECREATE")
h_rd.Write()
h_prompt.Write()
h_nprompt.Write()
h_bkg.Write()
h_pull_dist.Write()
c_pull_dist.Write()
c_fit.Write()
outfile.Close()

# Print results
tot = (n_prompt.getVal() + n_nprompt.getVal() + n_bkg.getVal())
print(f"\nχ² = {chi2:.2f}\n")
print("Prompt     :", n_prompt.getVal())
print("Non prompt :", n_nprompt.getVal())
print("Background :", n_bkg.getVal())
print("Total      :", tot)


# Features histograms after  bdt cut
# BDT regions

# Non-prompt enriched region
rd_nprompt = rd_df[(rd_df["model_output_2"] > 0.035) & (rd_df["model_output_2"] < 0.115)]
test_nprompt_nonpromptRegion = df_test[(df_test["label"] == 2) & (df_test["model_output_2"] > 0.035) & (df_test["model_output_2"] < 0.115)]


# Prompt enriched region
rd_prompt = rd_df[(rd_df["model_output_1"] > 0.9)]
test_prompt_promptRegion = df_test[(df_test["label"] == 1) & (df_test["model_output_1"] > 0.9)]

regions_p = {"prompt_region": (rd_prompt,test_prompt_promptRegion)}
regions_np = {"nprompt_region": (rd_nprompt,test_nprompt_nonpromptRegion)}
vars_to_plot = ['fMass','fDcaPosPV','fDcaNegPV','fDcaV0Tracks','fDcaV0PV','fCosPA','fRadius','fQtAP','fPt','fTpcNsigmaNeg','fTpcNsigmaPos']
outfile_features = ROOT.TFile(os.path.join(root_dir,"features_after_bdt_cut.root"),"RECREATE")


# Histograms for enriched prompt region
for region_name_p, (rd_sel_p, prompt_sel) in regions_p.items():
    for var in vars_to_plot:
        xmin_p = min(rd_sel_p[var].min(), prompt_sel[var].min())
        xmax_p = max(rd_sel_p[var].max(), prompt_sel[var].max())
        h_rd_p = ndarray2th1(rd_sel_p[var].to_numpy(),f"h_rd_{var}_{region_name_p}",Nbins, xmin_p, xmax_p)
        h_prompt = ndarray2th1(prompt_sel[var].to_numpy(),f"h_prompt_{var}_{region_name_p}",Nbins, xmin_p, xmax_p)
        h_rd_p.SetDirectory(0)
        h_prompt.SetDirectory(0)
        normalize(h_rd_p)
        normalize(h_prompt)
        ks_prompt = h_rd_p.KolmogorovTest(h_prompt, "M")
        print(f"{var} KS prompt = {ks_prompt:.5f}")
        h_rd_p.SetMarkerStyle(20)
        h_prompt.SetLineColor(ROOT.kRed + 1)
        h_prompt.SetMarkerColor(ROOT.kRed + 1)
        c_prompt = ROOT.TCanvas(f"c_prompt_{var}_{region_name_p}",f"{var} prompt",900,700)

        max_y_prompt = max(h_rd_p.GetMaximum(),h_prompt.GetMaximum())

        h_rd_p.SetMaximum(1.3 * max_y_prompt)
        h_rd_p.SetTitle(f"{var} : Real data vs Prompt after bdt cuts > 0.9")
        h_rd_p.GetXaxis().SetTitle(var)
        h_rd_p.GetYaxis().SetTitle("Normalized entries")
        h_rd_p.Draw("E")
        h_prompt.Draw("PE SAME")
        legend_prompt = ROOT.TLegend(0.60, 0.70, 0.88, 0.88)
        legend_prompt.AddEntry(h_rd_p, "Real data", "lep")
        legend_prompt.AddEntry(h_prompt, "Prompt MC", "lep")
        legend_prompt.SetBorderSize(0)
        legend_prompt.SetFillStyle(0)
        legend_prompt.Draw()

        stats_prompt = ROOT.TPaveText(0.60, 0.55, 0.88, 0.65, "NDC")
        stats_prompt.SetFillStyle(0)
        stats_prompt.SetBorderSize(0)
        stats_prompt.AddText(f"KS prompt = {ks_prompt:.4f}")
        stats_prompt.Draw()
    
        c_prompt.SaveAs(os.path.join(plot_dir,f"{var}_{region_name_p}_prompt.png"))
        outfile_features.cd()

        h_rd_p.Write()
        h_prompt.Write()
        c_prompt.Write()
        c_prompt.Close()
        del c_prompt
        del h_rd_p
        del h_prompt
        
        # Histograms for enriched nprompt region
for region_name_np, (rd_sel_np, nprompt_sel) in regions_np.items():            
    for var in vars_to_plot:                
        xmin_np = min(rd_sel_np[var].min(), nprompt_sel[var].min())
        xmax_np = max(rd_sel_np[var].max(), nprompt_sel[var].max())
        h_rd_np = ndarray2th1(rd_sel_np[var].to_numpy(),f"h_rd_{var}_{region_name_np}",Nbins, xmin_np,xmax_np)
        h_nprompt = ndarray2th1(nprompt_sel[var].to_numpy(),f"h_nprompt_{var}_{region_name_np}",Nbins, xmin_np, xmax_np)
        
        # Avoid ROOT ownership issues
        h_rd_np.SetDirectory(0)
        h_nprompt.SetDirectory(0)
        
        # Normalize
        normalize(h_rd_np)
        normalize(h_nprompt)

        # KS tests
        ks_nprompt = h_rd_np.KolmogorovTest(h_nprompt, "M")        
        print(f"{var} KS nonprompt = {ks_nprompt:.5f}")
        
        # Style
        h_rd_np.SetMarkerStyle(20)
        h_nprompt.SetLineColor(ROOT.kGreen + 1)
        h_nprompt.SetMarkerColor(ROOT.kGreen + 1)
        
        # Canvas nonprompt
        c_nprompt = ROOT.TCanvas(f"c_nprompt_{var}_{region_name_np}",f"{var} nonprompt",900,70)

        max_y_nprompt = max(h_rd_np.GetMaximum(),h_nprompt.GetMaximum())

        h_rd_np.SetMaximum(1.3 * max_y_nprompt)
        h_rd_np.SetTitle(f"{var} : Real data vs Non-prompt after bdt cuts 0.035 < bdt < 0.115")
        h_rd_np.GetXaxis().SetTitle(var)
        h_rd_np.GetYaxis().SetTitle("Normalized entries")
        h_rd_np.Draw("E")
        h_nprompt.Draw("PE SAME")

        legend_nprompt = ROOT.TLegend(0.60, 0.70, 0.88, 0.88)
        legend_nprompt.AddEntry(h_rd_np, "Real data", "lep")
        legend_nprompt.AddEntry(h_nprompt, "Non-prompt MC", "lep")
        legend_nprompt.SetBorderSize(0)
        legend_nprompt.SetFillStyle(0)
        legend_nprompt.Draw()

        stats_nprompt = ROOT.TPaveText(0.60, 0.55, 0.88, 0.65, "NDC")
        stats_nprompt.SetFillStyle(0)
        stats_nprompt.SetBorderSize(0)
        stats_nprompt.AddText(f"KS nonprompt = {ks_nprompt:.4f}")
        stats_nprompt.Draw()

        c_nprompt.SaveAs(os.path.join(plot_dir,f"{var}_{region_name_np}_nonprompt.png"))

        outfile_features.cd()
        
        #h_rd_nprompt = h_rd.Clone(f"h_rd_nprompt_{var}_{region_name}")
        #h_nprompt_clone = h_nprompt.Clone(f"h_nprompt_{var}_{region_name}")
        h_rd_np.Write()
        h_nprompt.Write()
        c_nprompt.Write()
        c_nprompt.Close()
        del c_nprompt
        del h_rd_np
        del h_nprompt
        
outfile_features.Close()




# Purity efficiency fom scan
cuts = np.linspace(0.0, 1.0, 200)
purity_prompt = []
purity_nprompt = []

eff_prompt = []
eff_nprompt = []
fom_prompt = []
fom_nprompt = []

# Scan over BDT cuts
for cut in cuts:

    # PROMPT
    df_prompt = df_test[df_test["model_output_1"] > cut]

    S_prompt = len(df_prompt[df_prompt["label"] == 1])
    B_prompt = len(df_prompt[df_prompt["label"] != 1])
    Ntot_prompt = S_prompt + B_prompt

    Nprompt_total = len(df_test[df_test["label"] == 1])
    if Ntot_prompt == 0:
        purity_prompt.append(0)
        eff_prompt.append(0)
        fom_prompt.append(0)
    else:
        purity_prompt.append(S_prompt / Ntot_prompt)
        eff_prompt.append(S_prompt / Nprompt_total)
        fom_prompt.append(S_prompt / np.sqrt(Ntot_prompt))
        
    # NON-PROMPT
    df_nprompt = df_test[df_test["model_output_2"] > cut]

    S_nprompt = len(df_nprompt[df_nprompt["label"] == 2])
    B_nprompt = len(df_nprompt[df_nprompt["label"] != 2])
    Ntot_nprompt = S_nprompt + B_nprompt
    Nnp_total = len(df_test[df_test["label"] == 2])
    if Ntot_nprompt == 0:
        purity_nprompt.append(0)
        eff_nprompt.append(0)
        fom_nprompt.append(0)
    else:
        purity_nprompt.append(S_nprompt / Ntot_nprompt)
        eff_nprompt.append(S_nprompt / Nnp_total)
        fom_nprompt.append(S_nprompt / np.sqrt(Ntot_nprompt))

# Best cuts
best_idx_prompt = np.argmax(fom_prompt)
best_cut_prompt = cuts[best_idx_prompt]
best_fom_prompt = fom_prompt[best_idx_prompt]
best_purity_prompt = purity_prompt[best_idx_prompt]
best_eff_prompt = eff_prompt[best_idx_prompt]

best_idx_nprompt = np.argmax(fom_nprompt)
best_cut_nprompt = cuts[best_idx_nprompt]
best_fom_nprompt = fom_nprompt[best_idx_nprompt]
best_purity_nprompt = purity_nprompt[best_idx_nprompt]
best_eff_nprompt = eff_nprompt[best_idx_nprompt]

# ROOT graphs
g_purity_prompt = ROOT.TGraph(len(cuts))
g_purity_nprompt = ROOT.TGraph(len(cuts))

g_eff_prompt = ROOT.TGraph(len(cuts))
g_eff_nprompt = ROOT.TGraph(len(cuts))

g_fom_prompt = ROOT.TGraph(len(cuts))
g_fom_nprompt = ROOT.TGraph(len(cuts))

for i in range(len(cuts)):

    # Purity
    g_purity_prompt.SetPoint(i,cuts[i],purity_prompt[i])
    g_purity_nprompt.SetPoint(i,cuts[i],purity_nprompt[i])

    # Efficiency
    g_eff_prompt.SetPoint(i,cuts[i],eff_prompt[i])
    g_eff_nprompt.SetPoint(i,cuts[i],eff_nprompt[i])

    # FOM
    g_fom_prompt.SetPoint(i,cuts[i],fom_prompt[i])
    g_fom_nprompt.SetPoint(i,cuts[i],fom_nprompt[i])

# Style
g_purity_prompt.SetLineColor(ROOT.kRed + 1)
g_eff_prompt.SetLineColor(ROOT.kBlue + 1)
g_fom_prompt.SetLineColor(ROOT.kBlack)

g_purity_nprompt.SetLineColor(ROOT.kGreen + 2)
g_eff_nprompt.SetLineColor(ROOT.kMagenta + 1)
g_fom_nprompt.SetLineColor(ROOT.kBlack)

graphs = [
    g_purity_prompt,
    g_eff_prompt,
    g_fom_prompt,
    g_purity_nprompt,
    g_eff_nprompt,
    g_fom_nprompt]

for g in graphs:
    g.SetLineWidth(2)

# Output ROOT file
outfile = ROOT.TFile(os.path.join(root_dir,"purity_scan.root"),"RECREATE")

# PROMPT CANVAS
c_prompt = ROOT.TCanvas("c_prompt_scan","Prompt optimization",900,700)

g_purity_prompt.SetTitle("Prompt optimization; BDT cut;Value")
g_purity_prompt.Draw("AL")
g_eff_prompt.Draw("L SAME")
g_fom_prompt.Draw("L SAME")

leg_prompt = ROOT.TLegend(0.60,0.65,0.88,0.88)
leg_prompt.AddEntry(g_purity_prompt,"Prompt purity","l")
leg_prompt.AddEntry(g_eff_prompt,"Prompt efficiency","l")
leg_prompt.AddEntry(g_fom_prompt,"Prompt FOM","l")
leg_prompt.Draw()

# Prompt TPave
pave_prompt = ROOT.TPaveText(0.15,0.65,0.45,0.88,"NDC")
pave_prompt.SetFillStyle(0)
pave_prompt.SetBorderSize(0)
pave_prompt.SetTextSize(0.03)
pave_prompt.AddText("Prompt study")

pave_prompt.AddText(f"Best cut = {best_cut_prompt:.3f}")
pave_prompt.AddText(f"Max FOM = {best_fom_prompt:.2f}")
pave_prompt.AddText(f"Purity = {best_purity_prompt:.3f}")
pave_prompt.AddText(f"Efficiency = {best_eff_prompt:.3f}")
pave_prompt.Draw()

# Save prompt
outfile.cd()
g_purity_prompt.Write("g_purity_prompt")
g_eff_prompt.Write("g_eff_prompt")
g_fom_prompt.Write("g_fom_prompt")
c_prompt.Write()
c_prompt.SaveAs(os.path.join(plot_dir,"prompt_optimization.png"))

# NON-PROMPT CANVAS
c_nprompt = ROOT.TCanvas("c_nonprompt_scan","Non-prompt optimization",900,700)

g_purity_nprompt.SetTitle("Non-prompt optimization;BDT cut;Value")
g_purity_nprompt.Draw("AL")
g_eff_nprompt.Draw("L SAME")
g_fom_nprompt.Draw("L SAME")

leg_nprompt = ROOT.TLegend(0.60,0.65,0.88,0.88)
leg_nprompt.AddEntry(g_purity_nprompt,"Non-prompt purity","l")
leg_nprompt.AddEntry(g_eff_nprompt,"Non-prompt efficiency","l")
leg_nprompt.AddEntry(g_fom_nprompt,"Non-prompt FOM","l")
leg_nprompt.Draw()

# Non-prompt TPave
pave_nprompt = ROOT.TPaveText(0.15,0.65,0.45,0.88,"NDC")
pave_nprompt.SetFillStyle(0)
pave_nprompt.SetBorderSize(0)
pave_nprompt.SetTextSize(0.03)
pave_nprompt.AddText("Non-prompt study")
pave_nprompt.AddText(f"Best cut = {best_cut_nprompt:.3f}")
pave_nprompt.AddText(f"Max FOM = {best_fom_nprompt:.2f}")
pave_nprompt.AddText(f"Purity = {best_purity_nprompt:.3f}")
pave_nprompt.AddText(f"Efficiency = {best_eff_nprompt:.3f}")
pave_nprompt.Draw()

# Save nonprompt
outfile.cd()
g_purity_nprompt.Write("g_purity_nonprompt")
g_eff_nprompt.Write("g_eff_nonprompt")
g_fom_nprompt.Write("g_fom_nonprompt")
c_nprompt.Write()
c_nprompt.SaveAs(os.path.join(plot_dir,"nonprompt_optimization.png"))

# Close ROOT file
outfile.Close()

# Print results
print("\n========== PROMPT ==========")
print(f"Best cut = " f"{best_cut_prompt:.3f}")
print(f"Max FOM = "  f"{best_fom_prompt:.3f}")
print(f"Purity = " f"{best_purity_prompt:.3f}")
print(f"Efficiency = " f"{best_eff_prompt:.3f}")


print("\n======= NON-PROMPT ========")
print(f"Best cut = " f"{best_cut_nprompt:.3f}")
print(f"Max FOM = "  f"{best_fom_nprompt:.3f}")
print(f"Purity = " f"{best_purity_nprompt:.3f}")
print(f"Efficiency = " f"{best_eff_nprompt:.3f}")

