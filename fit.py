######################
# fitting of the model to the data with RooFit, using the output score of the model as variable to fit, 
# and the templates of the prompt, non_prompt and background as elements of the fit, to verify if there
#  are differences between the data and the model
###################################################################
# downloading of the model trained
model_hdl = ModelHandler()
#model_hdl.load_model_handler("Risultati_Bkg_in_DatiMC/model_Bkg_In_DatiMC.pkl")
model_hdl.load_model_handler("Risultati_Bkg_in_DatiReali/model_Bkg_In_DatiReali.pkl")


# creation of the function ndarray2th1 for the convertion from numpy to TH1 histograms
def ndarray2th1(arr, name, nbins, xmin, xmax):
    arr = np.asarray(arr).flatten()
    hist = ROOT.TH1F(name, name, nbins, xmin, xmax)
    hist.SetDirectory(0)  # Disassociate the histogram from any ROOT file
    for x in arr:
        if np.isfinite(x) and xmin <= x <= xmax:
            hist.Fill(float(x))

    return hist        

# definition of the real data prediction for prompt, nprompt, and background
x_rd = Real_Data.get_subset('fRadius > 3 and fCosPA > 0.97', size = 1000000) # rd = real data, x_rd = real data sample
y_pred_rd = model_hdl.predict(x_rd, output_margin=False) # y_pred_rd = real data predicted
x_rd_bdt = np.array([y[1] for y in y_pred_rd]) # real data score prompt
#mask = x_rd_bdt > 0.8 # selection of the real data with a score prompt greater than 0.8, to have a sample of real data with a high purity of signal
#x_rd_bdt_cut = x_rd_bdt[mask] 

# definition of the test data predictions for prompt, non_prompt and background
y_pred_test = np.array(model_hdl.predict(x_test_rd, output_margin=False)) # y_pred_t = test data predicted
y_pred_test_prompt_out = y_pred_test[y_test_rd == 1, 1] # score prompt output for test data 
y_pred_test_nprompt_out = y_pred_test[y_test_rd == 2, 1] # score non prompt output for test data
y_pred_test_bkg_out = y_pred_test[y_test_rd == 0, 1] # score background output for test data


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

h_prompt.Smooth(1)
h_nprompt.Smooth(1)
h_bkg.Smooth(1)
h_rd.Smooth(1)

# normalization of the histograms to have the same scale for the fitting
def normalize(h):
    n = h.Integral()
    if n > 0:
        h.Scale(1.0 / n)

    for i in range(1, h.GetNbinsX() + 1):
        if h.GetBinContent(i) == 0:     
            h.SetBinContent(i, 1e-6)    
normalize(h_prompt)
normalize(h_nprompt)
normalize(h_bkg)
normalize(h_rd)


print("Entries prompt template :", h_prompt.Integral())
print("Entries nprompt template:", h_nprompt.Integral())
print("Entries bkg template    :", h_bkg.Integral())
print("Entries data            :", h_rd.Integral())

# quantity values of prompt, non_prompt and background in 0 - 1e6
n_prompt = ROOT.RooRealVar("n_prompt", "n_prompt", 0.6*len(x_rd_bdt), 0, len(x_rd_bdt))
n_nprompt = ROOT.RooRealVar("n_nprompt", "n_nprompt", 0.3*len(x_rd_bdt), 0, len(x_rd_bdt))
n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", 0.1*len(x_rd_bdt), 0, len(x_rd_bdt))


# creation of the model
model = ROOT.RooAddPdf(
    "model", "",
    ROOT.RooArgList(pdf_prompt, pdf_nprompt, pdf_bkg),
    ROOT.RooArgList(n_prompt, n_nprompt, n_bkg))

# fit of the model
fit_result = model.fitTo(bdt_rd, ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True))
fit_result.Print()

ROOT.gROOT.SetBatch(True) 

# canvas
c = ROOT.TCanvas(f"data", "", 800, 700)
c.SetLeftMargin(0.15)
c.SetBottomMargin(0.12)

# frame with title and axis
frame = bdt_out.frame(ROOT.RooFit.Title("BDT output fit for prompt"))
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

# chi2 and chi2/ndf
n_fit_param = 3
chi2 = frame.chiSquare("model", "real data", n_fit_param)
ndf = Nbins_output_score - n_fit_param
print("chi2 =", chi2)
print("chi2/ndf =", chi2/ndf)

# pave with the fit results and the chi2/ndf
pave = ROOT.TPaveText(0.60, 0.65, 0.88, 0.88, "NDC")
pave.AddText(f"#chi^2 / ndf = {chi2/ndf:.2f}")
pave.SetFillStyle(0)
pave.SetBorderSize(0)
pave.SetTextAlign(12)
pave.AddText(f"n_prompt = ({n_prompt.getVal():,.2f} ± {n_prompt.getError():,.2f})")
pave.AddText(f"n_nprompt = ({n_nprompt.getVal():,.2f} ± {n_nprompt.getError():,.2f})")
pave.AddText(f"n_bkg = ({n_bkg.getVal():,.2f} ± {n_bkg.getError():,.2f})")
pave.Draw()


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
pave.Draw()
c.Modified()
c.Update()

# create and save root file
file_out = ROOT.TFile("fit_bdt_p.root", "RECREATE")
dir_bdt_output = file_out.mkdir("RealData_TestData")
dir_bdt_output.cd()

h_rd.Write("h_rd")
h_prompt.Write("h_prompt")
h_nprompt.Write("h_nprompt")
h_bkg.Write("h_bkg")

c.Write("canvas_fit")
file_out.Write()
file_out.Close()    
