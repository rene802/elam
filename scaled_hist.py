########################################################################
# Process to select prompt and non_prompt cadidates using BDT score cuts
# Integration of the templates of the prompt, non_prompt and background 
# according the bdt cut sidband BDT score > 0.9 and BDT score < 0.15
########################################################################

#############################################################
# Define the bin ranges for the prompt and non_prompt regions
###############################################################
binp_min = h_prompt.FindBin(0.9)
binp_max = h_prompt.FindBin(1.0)

binnp_min = h_prompt.FindBin(0.0)
binnp_max = h_prompt.FindBin(0.15)

###################################################################################
# Integration of the templates in the defined bin ranges to get the expected number 
# of candidates in the prompt and non_prompt regions
####################################################################################
I_right_p = h_prompt.Integral(binp_min, binp_max)
I_right_np = h_nprompt.Integral(binp_min, binp_max)
I_right_bkg = h_bkg.Integral(binp_min, binp_max)

I_left_p = h_prompt.Integral(binnp_min, binnp_max)
I_left_np = h_nprompt.Integral(binnp_min, binnp_max)
I_left_bkg = h_bkg.Integral(binnp_min, binnp_max)

#########################################################################################
# Number of candidates expected in the following BDT score range bdt > 0.9 and bdt < 1.0
#########################################################################################
N_p_right = n_prompt.getVal() * I_right_p
N_np_right = n_nprompt.getVal() * I_right_np
N_bkg_right = n_bkg.getVal() * I_right_bkg

N_p_left = n_prompt.getVal() * I_left_p
N_np_left = n_nprompt.getVal() * I_left_np
N_bkg_left = n_bkg.getVal() * I_left_bkg

##################################################################################################
# Purity of the prompt and non_prompt candidates in the following BDT score range > 0.9 and < 0.15
##################################################################################################
purity_prompt = N_p_right / (N_p_right + N_np_right + N_bkg_right)
purity_nprompt = N_np_left / (N_np_left + N_p_left + N_bkg_left)

#################################################################################
# Cuts on the real data to select candidates in the prompt and non_prompt regions
#################################################################################
prompt_df = rd_df[rd_df["model_output_1"] > 0.9]
non_prompt_df = rd_df[rd_df["model_output_2"] < 0.15]

############################################
# Templates in BDT cut regions
# with ROOT file saving
############################################

############################################################
# Right region : BDT > 0.9 for prompt enriched region
##############################################################

# Clone histograms for scaling
h_promptR_scaled = h_prompt.Clone("h_promptR_scaled")
h_npromptR_scaled = h_nprompt.Clone("h_npromptR_scaled")
h_bkgR_scaled = h_bkg.Clone("h_bkgR_scaled")

# Scale with fitted yields
h_promptR_scaled.Scale(N_p_right)
h_npromptR_scaled.Scale(N_np_right)
h_bkgR_scaled.Scale(N_bkg_right)

# Canvas
c_right = ROOT.TCanvas("c_right", "Prompt region", 800, 600)

# Restrict x-axis
h_promptR_scaled.GetXaxis().SetRangeUser(0.9, 1.0)
h_npromptR_scaled.GetXaxis().SetRangeUser(0.9, 1.0)
h_bkgR_scaled.GetXaxis().SetRangeUser(0.9, 1.0)

# Style
h_promptR_scaled.SetLineColor(ROOT.kRed)
h_promptR_scaled.SetLineWidth(3)

h_npromptR_scaled.SetLineColor(ROOT.kBlue)
h_npromptR_scaled.SetLineWidth(3)

h_bkgR_scaled.SetLineColor(ROOT.kGreen + 2)
h_bkgR_scaled.SetLineWidth(3)

# Maximum
max_right = max(h_promptR_scaled.GetMaximum(), h_npromptR_scaled.GetMaximum(), h_bkgR_scaled.GetMaximum())
h_promptR_scaled.SetMaximum(max_right * 1.3)

# Titles
h_promptR_scaled.SetTitle("Prompt enriched region : BDT > 0.9")
h_promptR_scaled.GetXaxis().SetTitle("BDT score")
h_promptR_scaled.GetYaxis().SetTitle("Expected candidates")

# Draw
h_promptR_scaled.Draw("HIST")
h_npromptR_scaled.Draw("HIST SAME")
h_bkgR_scaled.Draw("HIST SAME")

# Legend
leg_right = ROOT.TLegend(0.60, 0.70, 0.88, 0.88)
leg_right.AddEntry(h_promptR_scaled, "Prompt", "l")
leg_right.AddEntry(h_npromptR_scaled, "Non-prompt", "l")
leg_right.AddEntry(h_bkgR_scaled, "Background", "l")
leg_right.Draw()

# Save image
c_right.SaveAs("prompt_regionR_scaled.png")

#########################################################
# Left region : BDT < 0.15 for non-prompt enriched region
#########################################################

# Clone histograms for scaling
h_promptL_scaled = h_prompt.Clone("h_promptL_scaled")
h_npromptL_scaled = h_nprompt.Clone("h_npromptL_scaled")
h_bkgL_scaled = h_bkg.Clone("h_bkgL_scaled")

# Scale with fit yields
h_promptL_scaled.Scale(N_p_left)
h_npromptL_scaled.Scale(N_np_left)
h_bkgL_scaled.Scale(N_bkg_left)

# Canvas
c_left = ROOT.TCanvas("c_left", "Non-prompt region", 800, 600)

# Restrict x-axis
h_promptL_scaled.GetXaxis().SetRangeUser(0.0, 0.15)
h_npromptL_scaled.GetXaxis().SetRangeUser(0.0, 0.15)
h_bkgL_scaled.GetXaxis().SetRangeUser(0.0, 0.15)

# Style
h_promptL_scaled.SetLineColor(ROOT.kRed)
h_promptL_scaled.SetLineWidth(3)

h_npromptL_scaled.SetLineColor(ROOT.kBlue)
h_npromptL_scaled.SetLineWidth(3)

h_bkgL_scaled.SetLineColor(ROOT.kGreen + 2)
h_bkgL_scaled.SetLineWidth(3)

# Maximum
max_left = max(h_promptL_scaled.GetMaximum(),h_npromptL_scaled.GetMaximum(),h_bkgL_scaled.GetMaximum())
h_promptL_scaled.SetMaximum(max_left * 1.3)

# Titles
h_promptL_scaled.SetTitle("Rejected prompt region : BDT < 0.15")
h_promptL_scaled.GetXaxis().SetTitle("BDT score")
h_promptL_scaled.GetYaxis().SetTitle("Expected candidates")

# Draw
h_promptL_scaled.Draw("HIST")
h_npromptL_scaled.Draw("HIST SAME")
h_bkgL_scaled.Draw("HIST SAME")

# Legend
leg_left = ROOT.TLegend(0.60, 0.70, 0.88, 0.88)

leg_left.AddEntry(h_promptL_scaled, "Prompt", "l")
leg_left.AddEntry(h_npromptL_scaled, "Non-prompt", "l")
leg_left.AddEntry(h_bkgL_scaled, "Background", "l")
leg_left.Draw()

# Save image
c_left.SaveAs("nonprompt_region_scaled.png")

############################################
# Save all ROOT objects
############################################
outfile_regions = ROOT.TFile("bdt_regions_templates.root", "RECREATE")

############################################
# Write histograms
############################################

# Right region
h_promptR_scaled.Write()
h_npromptR_scaled.Write()
h_bkgR_scaled.Write()

# Left region
h_promptL_scaled.Write()
h_npromptL_scaled.Write()
h_bkgL_scaled.Write()

############################################
# Write canvases
############################################
c_right.Write()
c_left.Write()

############################################
# Close ROOT file
############################################
outfile_regions.Close()

print("\nROOT file saved : bdt_regions_templates.root")
print("PNG images saved :")
print(" - prompt_region_scaled.png")
print(" - nonprompt_region_scaled.png")
