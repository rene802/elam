import pickle
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from config import model_dir, plot_dir, root_dir, log_dir

#MC_Data = TreeHandler('AODMC.root', 'DF_2261906078815382/O2mclambdatableml')
#Real_Data = TreeHandler('AODDat.root', 'DF_2261906085717504/O2lambdatableml')

# Data loading in tree handlers
MC_Data = TreeHandler('AO2D_687783_MC_merged.root', 'DF_2364001568637363/O2mclambdatableml')
Real_Data = TreeHandler('AO2D_692473_merged.root', 'DF_2364053441300768/O2lambdatableml')

# Split of real data in two subsets for training and fitting
df_real = Real_Data.get_data_frame()
df_bkg_train, df_bkg_fit = train_test_split(df_real, test_size=0.5, random_state=42)

# Creation of tree handlers for the background training and fitting subsets
bkg_train = TreeHandler()
bkg_train.set_data_frame(df_bkg_train)

bkg_fit = TreeHandler()
bkg_fit.set_data_frame(df_bkg_fit)

# Selection of prompt, non-prompt and background candidates for training the model
Prompt = MC_Data.get_subset('fIsReco == 1 and fCosPA > 0.97 and fRadius > 3 and (fPDGCode == 3122 or fPDGCode == -3122) and (abs(fPDGCodeMother) != 3312 and abs(fPDGCodeMother) != 3322 and abs(fPDGCodeMother) != 3334)')
Non_Prompt = MC_Data.get_subset('fIsReco == 1 and fCosPA > 0.97 and fRadius > 3 and  (fPDGCode == 3122 or fPDGCode == -3122) and (abs(fPDGCodeMother) == 3312 or abs(fPDGCodeMother) == 3322 or abs(fPDGCodeMother) == 3334)')
Background = bkg_train.get_subset('fRadius > 3 and fCosPA > 0.97 and (fMass < 1.098 or fMass > 1.145)', size = None)

train_test_data = train_test_generator([Background, Prompt, Non_Prompt],[0, 1, 2],test_size=0.5,random_state=42)
x_train, y_train, x_test, y_test = train_test_data

len(Prompt), len(Non_Prompt),len(Background)


# Application of pT cut on the three categories for training and fitting
pt_cut = "fPt > 1 and fPt < 4"
Prompt = Prompt.get_subset(pt_cut)
Non_Prompt = Non_Prompt.get_subset(pt_cut)
Background = Background.get_subset(pt_cut)
bkg_fit = bkg_fit.get_subset(pt_cut)

# Equalization of the number of events in the three categories for training
n_min = min(len(Background), len(Prompt), len(Non_Prompt))
print("Min events:", n_min)

Background = Background.get_subset(size=n_min)
Prompt = Prompt.get_subset(size=n_min )
Non_Prompt = Non_Prompt.get_subset(size=n_min )

# Print the number of events in each category after cuts and equalization events in each category after cuts and equalization
print(f"Length of Prompt: {len(Prompt)}")
print(f"Length of Non_Prompt: {len(Non_Prompt)}")
print(f"Length of Background: {len(Background)}")

# Save the training and test data for future use
output_file = os.path.join(model_dir, "train_testp_data.pkl")
pickle.dump(train_test_data, open(output_file, "wb"))
