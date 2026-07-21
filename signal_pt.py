import os
import numpy as np
import pandas as pd
import ROOT
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator
from config_analysis import *


#####################
# GENERATED LAMBDA
#####################
# load data 
MC_Data = TreeHandler('AO2D_687783_MC_merged.root', 'DF_2364001568637363/O2mclambdatableml')
df_mc = MC_Data.get_data_frame()
print(df_mc.columns.tolist())

# selection of generated lambda
GenLambda = MC_Data.get_subset("abs(fPDGCode) == 3122 and (abs(fPDGCodeMother) != 3312 and abs(fPDGCodeMother) != 3322 and abs(fPDGCodeMother) != 3334)")
df_gen = GenLambda.get_data_frame()


# Efficiency storage
eff_vals = []
eff_errs = []
corr_vals = []
corr_errs = []
Ngen_vals = []
Nreco_vals = []
yield_vals = []
yield_err_vals = []

# selection of generated lambda
for ptmin, ptmax in pt_bins:
    N_gen = len(df_gen[(df_gen["fGenPt"] >= ptmin) & (df_gen["fGenPt"] < ptmax)])
    Ngen_vals.append(N_gen)
    print(ptmin, ptmax, N_gen)


#####################################################################################################################################################################################################

########################
# RECONSTRUCTED LAMBDA
########################

# imput parquet
test_parquet = os.path.join(model_dir,"TestData_apply_pt_sm.parquet.gzip")
print("\nLoading parquet...\n")
df = pd.read_parquet(test_parquet)

# array for final spectrum
x_vals = []
y_vals = []
x_errs = []
y_errs = []


# ouput file
out = ROOT.TFile(os.path.join(root_dir, "mass_spectrum_roofit.root"),"RECREATE")

# real data from parquet file  
real_parquet = os.path.join(model_dir,"real_16NewData.parquet.gzip")
print("\nLoading parquet...\n")
df_data = pd.read_parquet(real_parquet)
print(df_data.columns.tolist())

# loop over pt bins
for ptmin, ptmax in pt_bins:
    print("\n====================================")
    print(f"Processing pT bin: {ptmin} - {ptmax}")
    print(f"BDT cut = {bdt_cut}")
    print("====================================")

    # Selection
    df_bin = df[(df["fPt"] >= ptmin) & (df["fPt"] < ptmax) & (df["model_output_1"] > bdt_cut)]
    masses = df_bin["fMass"].to_numpy()

    # Mass histogram
    h_mass = ROOT.TH1F(f"h_mass_{ptmin}_{ptmax}",f"{ptmin} < p_{{T}} < {ptmax};M (GeV/c^{{2}});Counts",Nbins, *mass_range)
    for m in masses:
        if np.isfinite(m):
            h_mass.Fill(m)
    
    # variable masse
    mass_mc = ROOT.RooRealVar(f"mass_mc_{ptmin}_{ptmax}","M(p#pi)",mass_range[0],mass_range[1])

    # conversion histogram -> RooDataHist
    roo_data_mc = ROOT.RooDataHist(f"roo_data_mc_{ptmin}_{ptmax}","",ROOT.RooArgList(mass_mc),h_mass)


    # SIGNAL PDF
    mean = ROOT.RooRealVar(f"mean_{ptmin}_{ptmax}","mean",1.1157,1.113,1.118)
    sigma = ROOT.RooRealVar(f"sigma_{ptmin}_{ptmax}","sigma",0.0012,0.0005,0.005)

    alphaL = ROOT.RooRealVar(f"alphaL_{ptmin}_{ptmax}","alphaL",1.5,0.2,10.)
    alphaR = ROOT.RooRealVar(f"alphaR_{ptmin}_{ptmax}","alphaR",2.0,0.2,10.)

    nL = ROOT.RooRealVar(f"nL_{ptmin}_{ptmax}","nL",5.,0.5,100.)
    nR = ROOT.RooRealVar(f"nR_{ptmin}_{ptmax}","nR",5.,0.5,100.)

    signal_pdf = ROOT.RooCrystalBall(f"signal_pdf_{ptmin}_{ptmax}","",mass_mc,mean,sigma,alphaL,nL,alphaR,nR)


    # BACKGROUND PDF (Exponential)
    tau = ROOT.RooRealVar(f"tau_{ptmin}_{ptmax}","tau",-30,-100,-1)
    background = ROOT.RooExponential(f"background_{ptmin}_{ptmax}","",mass_mc,tau)

    # YIELDS
    Nsig = ROOT.RooRealVar(f"Nsig_{ptmin}_{ptmax}","signal yield",0.8*h_mass.Integral(),0,10*h_mass.Integral())
    Nbkg = ROOT.RooRealVar(f"Nbkg_{ptmin}_{ptmax}","background yield",0.2*h_mass.Integral(),0,10*h_mass.Integral())

    # TOTAL MODEL
    model = ROOT.RooAddPdf(f"model_{ptmin}_{ptmax}","signal+bkg",ROOT.RooArgList(signal_pdf, background),ROOT.RooArgList(Nsig, Nbkg))

    # EXTENDED FIT
    fit_result = model.fitTo(roo_data_mc,ROOT.RooFit.Save(),ROOT.RooFit.Extended(True),ROOT.RooFit.PrintLevel(-1))
    
    mc_mean = mean.getVal()
    mc_sigma = sigma.getVal()
    mc_mean_err = mean.getError()
    mc_sigma_err = sigma.getError()
    
    mc_alphaL = alphaL.getVal()
    mc_alphaR = alphaR.getVal()
    
    mc_nL = nL.getVal()  
    mc_nR = nR.getVal()

    print("Saved MC shape:")
    print(mc_mean, mc_sigma)
    

    # RESULTS
    signal_yield = Nsig.getVal()
    signal_yield_err = Nsig.getError()
    print()
    print("===== ROOFIT =====")
    print("Yield =", signal_yield)
    print("Yield err =", signal_yield_err)
    print("mean =", mean.getVal())
    print("sigma =", sigma.getVal())
    print("==================")
    print()

    # SIGNAL YIELD FROM ROOFIT
    print()
    print("----- SIGNAL EXTRACTION (ROOFIT) -----")
    print(f"mu            = {mean.getVal():.6f}")
    print(f"sigma         = {sigma.getVal():.6f}")
    print(f"Nsignal       = {signal_yield:.1f} ± {signal_yield_err:.1f}")
    print(f"Nbackground   = {Nbkg.getVal():.1f}")
    print("--------------------------------------")
    print()

    # EFFICIENCY
    N_gen = len(df_gen[(df_gen["fGenPt"] >= ptmin) & (df_gen["fGenPt"] < ptmax)])
    N_gen_test = 0.5 * N_gen
    efficiency = signal_yield / N_gen_test

    if efficiency > 0:
        efficiency_error = signal_yield_err / N_gen_test
    else:
        efficiency_error = 0

    # Store efficiency information

    eff_vals.append(efficiency)
    eff_errs.append(efficiency_error)

    Nreco_vals.append(signal_yield)

    if efficiency > 0:
        correction = 1.0 / efficiency
        correction_error = (efficiency_error / efficiency**2)
    else:
        correction = 0
        correction_error = 0

    corr_vals.append(correction)
    corr_errs.append(correction_error)

    print("----- EFFICIENCY -----")
    print("Ngen =", N_gen)
    print("Nreco =", signal_yield)
    print("epsilon =", efficiency)
    print("---------------------")
    

    # Fill dNraw/dpT spectrum
    pt_center = 0.5 * (ptmin + ptmax)
    
    # half bin width for x error
    pt_width = 0.5 * (ptmax - ptmin)

    # full bin width ΔpT
    delta_pt = ptmax - ptmin
    
    # selection of lambda candidates in data
    df_bin_data = df_data[(df_data["fPt"] >= ptmin) & (df_data["fPt"] < ptmax) & (df_data["model_output_1"] > bdt_cut)]
    mass_raw = df_bin_data["fMass"].to_numpy()
    # histogram of mass_raw
    h_mass_raw = ROOT.TH1F(f"h_mass_raw_{ptmin}_{ptmax}",f"{ptmin} < p_{{T}} < {ptmax};M (GeV/c^{{2}});Counts",Nbins, *mass_range)
    for m in mass_raw:
        if np.isfinite(m):
            h_mass_raw.Fill(m)

    # ROOFIT SIGNAL EXTRACTION ON REAL DATA
    mass = ROOT.RooRealVar(
    f"mass_{ptmin}_{ptmax}",
    "mass",
    mass_range[0],
    mass_range[1])

    roo_data_raw = ROOT.RooDataHist(f"roo_data_raw_{ptmin}_{ptmax}","",ROOT.RooArgList(mass),h_mass_raw)

    # Signal PDF
    mean_raw = ROOT.RooRealVar(f"mean_raw_{ptmin}_{ptmax}","mean",1.1157,1.112,1.118)
    sigma_raw = ROOT.RooRealVar(f"sigma_raw_{ptmin}_{ptmax}","sigma",0.0012,0.0005,0.005)

    alphaL_raw = ROOT.RooRealVar(f"alphaL_raw_{ptmin}_{ptmax}","",alphaL.getVal(), 0.1,20)
    nL_raw = ROOT.RooRealVar(f"nL_raw_{ptmin}_{ptmax}","",nL.getVal(), 0.1,100)

    alphaR_raw = ROOT.RooRealVar(f"alphaR_raw_{ptmin}_{ptmax}","",alphaR.getVal(), 0.1,20)
    nR_raw = ROOT.RooRealVar(f"nR_raw_{ptmin}_{ptmax}","",nR.getVal(), 0.1,100)

    signal_pdf_raw = ROOT.RooCrystalBall(f"signal_pdf_raw_{ptmin}_{ptmax}","",mass,mean_raw,sigma_raw,alphaL_raw,nL_raw,alphaR_raw,nR_raw)
    
    # use the signal shape got on the MC 
    mean_raw.setVal(mc_mean)
    sigma_raw.setVal(mc_sigma)

    mean_raw.setRange(mc_mean-0.0003,mc_mean+0.0003)
    mean_raw.setConstant(False)

    alphaL_raw.setVal(mc_alphaL)
    alphaR_raw.setVal(mc_alphaR)

    alphaL_raw.setConstant(True)
    alphaR_raw.setConstant(True)
    
    nL_raw.setVal(mc_nL)
    nR_raw.setVal(mc_nR)
    
    nL_raw.setConstant(True)
    nR_raw.setConstant(True)

    sigma_raw.setRange(0.8*mc_sigma, 1.2*mc_sigma)
    sigma_raw.setConstant(False)

    # Background PDF (Exponential)
    tau_raw = ROOT.RooRealVar(f"tau_raw_{ptmin}_{ptmax}","tau",-30,-100,-1)
    background_pdf_raw = ROOT.RooExponential(f"background_pdf_raw_{ptmin}_{ptmax}","",mass,tau_raw)

    # Yields
    nentries_raw = h_mass_raw.Integral()

    nsig_raw = ROOT.RooRealVar(f"nsig_raw_{ptmin}_{ptmax}","signal yield",0.8*nentries_raw,0,2*nentries_raw)
    nbkg_raw = ROOT.RooRealVar(f"nbkg_raw_{ptmin}_{ptmax}","background yield",0.2*nentries_raw,0,2*nentries_raw)

    model_raw = ROOT.RooAddPdf(f"model_raw_{ptmin}_{ptmax}","",ROOT.RooArgList(signal_pdf_raw,background_pdf_raw),ROOT.RooArgList(nsig_raw,nbkg_raw))
    fit_result_raw = model_raw.fitTo(roo_data_raw,ROOT.RooFit.Save(),ROOT.RooFit.Extended(True),ROOT.RooFit.PrintLevel(-1))

    # Extract signal yield
    N_raw_data = nsig_raw.getVal()
    N_raw_data_err = nsig_raw.getError()
    print()
    print("----- DATA SIGNAL EXTRACTION -----")
    print(f"Nsignal = {N_raw_data:.0f} ± {N_raw_data_err:.0f}")
    print(f"Nbkg    = {nbkg_raw.getVal():.0f}")
    print(f"mu      = {mean_raw.getVal():.6f}")
    print(f"sigma   = {sigma_raw.getVal():.6f}")
    print("----------------------------------")
    print()

    # correction and error correction
    if N_raw_data > 0 and efficiency > 0:        
        yield_density = (N_raw_data /delta_pt /efficiency)
        yield_density_err = (yield_density * np.sqrt((N_raw_data_err/N_raw_data)**2 + (efficiency_error/efficiency)**2))
    else:
        yield_density_err = 0
        yield_density = 0
        
    # holding 
    x_vals.append(pt_center)
    y_vals.append(yield_density)
    x_errs.append(pt_width)
    y_errs.append(yield_density_err)

    yield_vals.append(yield_density)
    yield_err_vals.append(yield_density_err)

    # Draw mass histogram
    c_mass = ROOT.TCanvas(f"c_mass_{ptmin}_{ptmax}","",800,600)

    ROOT.gStyle.SetOptStat(0)
    ROOT.TGaxis.SetMaxDigits(3)

    frame = mass.frame(ROOT.RooFit.Title(f"#Lambda candidates, {ptmin:.1f} < p_{{T}} < {ptmax:.1f} GeV/c"))
    roo_data_raw.plotOn(frame)
    model_raw.plotOn(frame)
    model_raw.plotOn(frame,ROOT.RooFit.Components(background_pdf_raw.GetName()),ROOT.RooFit.LineStyle(ROOT.kDashed))
    model_raw.plotOn(frame,ROOT.RooFit.Components(signal_pdf_raw.GetName()),
    ROOT.RooFit.LineColor(ROOT.kRed),
    ROOT.RooFit.LineStyle(ROOT.kDashed))

    # objets displied in RooPlot
    dataCurve = frame.getObject(0)
    totalFit = frame.getObject(1)
    backgroundFit = frame.getObject(2)
    signalFit = frame.getObject(3)
    c = ROOT.TCanvas(f"c_raw_{ptmin}_{ptmax}","",800,600)

    frame.Draw()
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)

    latex.DrawLatex(0.18,0.86,"This Thesis")
    latex.DrawLatex(0.58,0.84,f"#mu = {mean_raw.getVal():.5f} #pm {mean_raw.getError():.5f} GeV/c^{{2}}")
    latex.DrawLatex(0.58,0.78,f"#sigma = {sigma_raw.getVal():.5f} #pm {sigma_raw.getError():.5f} GeV/c^{{2}}")
    latex.DrawLatex(0.58,0.72,f"N_{{sig}} = {N_raw_data:.0f} #pm {N_raw_data_err:.0f}")
    latex.DrawLatex(0.58,0.66,f"N_{{bkg}} = {nbkg_raw.getVal():.0f} #pm {nbkg_raw.getError():.0f}")


    # Legend
    legend = ROOT.TLegend(0.14,0.56,0.40,0.78)

    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.03)

    legend.AddEntry(dataCurve,"Data","pe")
    legend.AddEntry(totalFit,"fit","l")
    legend.AddEntry(backgroundFit,"Background","l")
    legend.AddEntry(signalFit,"Signal","l")

    legend.Draw()

    out.cd()
    c.Write()
    c.SaveAs(os.path.join(plot_dir,f"mass_roofit_{ptmin}_{ptmax}.png"))

# FINAL PT SPECTRUM
x_vals = np.array(x_vals, dtype=np.float64)
y_vals = np.array(y_vals, dtype=np.float64)
x_errs = np.array(x_errs, dtype=np.float64)
y_errs = np.array(y_errs, dtype=np.float64)


graph = ROOT.TGraphErrors(len(x_vals),x_vals,y_vals,x_errs,y_errs)
graph.SetTitle(";p_{T} (GeV/c);dN_{corr}/dp_{T} (GeV/c)^{-1}")

graph.SetMarkerStyle(20)
graph.SetMarkerSize(1.3)
graph.SetLineWidth(2)




# DRAW FINAL SPECTRUM
c_spectrum = ROOT.TCanvas("c_spectrum","",900,700)

#ROOT.gPad.SetLogy()
graph.Draw("AP")

legend_pt = ROOT.TLegend(0.55,0.75,0.88,0.88)
legend_pt.SetBorderSize(0)
legend_pt.SetFillStyle(0)

legend_pt.AddEntry(graph,f"BDT > {bdt_cut}","p")

latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextSize(0.045)
latex.DrawLatex(0.18,0.85,"This Thesis")

legend_pt.Draw()

out.cd()
graph.Write()
c_spectrum.Write()
c_spectrum.SaveAs(os.path.join(plot_dir,"pt_spectrum_roofit.png"))

# SAVE EFFICIENCY TABLE
eff_table = pd.DataFrame({
    "pT_min": [x[0] for x in pt_bins],
    "pT_max": [x[1] for x in pt_bins],
    "pT_center": x_vals,
    "N_generated": Ngen_vals,
    "N_reconstructed": Nreco_vals,
    "Efficiency": eff_vals,
    "Efficiency_error": eff_errs,
    "Correction_factor": corr_vals,
    "Correction_error": corr_errs,
    "(1/ε) dNraw/dpT": yield_vals,
    "(1/ε) dNraw/dpT_error": yield_err_vals})

# SAVE TABLE AS PDF
pdf_file = os.path.join(plot_dir, "Lambda_efficiency_table.pdf")
doc = SimpleDocTemplate(pdf_file,pagesize=(29.7*cm, 21.0*cm))

styles = getSampleStyleSheet()
elements = []

title = Paragraph("<b>Lambda reconstruction efficiency as a function of transverse momentum</b>",styles["Heading2"])
elements.append(title)

# Colomun
table_data = [["pT min","pT max","pT center","Ngen","Nreco","Efficiency","Err(Eff)","1/Eff","Err(1/Eff)","(1/ε) dNraw/dpT","(1/ε) dNraw/dpT_error"]]

# Lignes
for _, row in eff_table.iterrows():
    table_data.append([
        f"{row['pT_min']:.1f}",
        f"{row['pT_max']:.1f}",
        f"{row['pT_center']:.2f}",
        f"{int(row['N_generated'])}",
        f"{int(row['N_reconstructed'])}",
        f"{row['Efficiency']:.5f}",
        f"{row['Efficiency_error']:.5f}",
        f"{row['Correction_factor']:.3f}",
        f"{row['Correction_error']:.3f}",
        f"{row['(1/ε) dNraw/dpT']:.8f}",
        f"{row['(1/ε) dNraw/dpT_error']:.8f}"
    ])

table = Table(table_data)

table.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
    ('TEXTCOLOR',(0,0),(-1,0),colors.black),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
    ('FONTSIZE',(0,0),(-1,-1),10),
    ('ALIGN',(0,0),(-1,-1),'CENTER'),
    ('GRID',(0,0),(-1,-1),0.5,colors.black),
    ('BOTTOMPADDING',(0,0),(-1,0),8),
    ('BACKGROUND',(0,1),(-1,-1),colors.white)]))

elements.append(table)
doc.build(elements)

print(f"\nPDF table saved in:\n{pdf_file}")
eff_table.to_latex(os.path.join(plot_dir,"Lambda_efficiency_table.tex"),index=False,float_format="%.5f",caption="Reconstruction efficiency as function of transverse momentum",label="tab:efficiency")
print("\nEfficiency table saved:")
print(eff_table)

out.Close()

print("\nDone. Spectrum extracted with RooFit.")
