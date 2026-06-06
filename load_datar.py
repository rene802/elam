import pickle
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from config import model_dir, plot_dir, root_dir, log_dir


MC_Data = TreeHandler('AODMC.root', 'DF_2261906078815382/O2mclambdatableml')
Real_Data = TreeHandler('AODDat.root', 'DF_2261906085717504/O2lambdatableml')

df_real = Real_Data.get_data_frame()
df_bkg_train, df_bkg_fit = train_test_split(df_real, test_size=0.5, random_state=42)

bkg_train = TreeHandler()
bkg_train.set_data_frame(df_bkg_train)

bkg_fit = TreeHandler()
bkg_fit.set_data_frame(df_bkg_fit)


Prompt = MC_Data.get_subset('fIsReco == 1 and fCosPA > 0.97 and fRadius > 3 and (fPDGCode == 3122 or fPDGCode == -3122) and (abs(fPDGCodeMother) != 3312 and abs(fPDGCodeMother) != 3322 and abs(fPDGCodeMother) != 3334)')
Non_Prompt = MC_Data.get_subset('fIsReco == 1 and fCosPA > 0.97 and fRadius > 3 and  (fPDGCode == 3122 or fPDGCode == -3122) and (abs(fPDGCodeMother) == 3312 or abs(fPDGCodeMother) == 3322 or abs(fPDGCodeMother) == 3334)')
Background = bkg_train.get_subset('fRadius > 3 and fCosPA > 0.97 and (fMass < 1.098 or fMass > 1.145)', size = None)

train_test_data = train_test_generator([Background, Prompt, Non_Prompt],[0, 1, 2],test_size=0.5,random_state=42)
x_train, y_train, x_test, y_test = train_test_data

len(Prompt), len(Non_Prompt),len(Background)



pt_cut = "fPt > 1 and fPt < 5"
Prompt = Prompt.get_subset(pt_cut)
Non_Prompt = Non_Prompt.get_subset(pt_cut)
Background = Background.get_subset(pt_cut)
bkg_fit = bkg_fit.get_subset(pt_cut)

n_min = min(len(Background), len(Prompt), len(Non_Prompt))
print("Min events:", n_min)

Background = Background.get_subset(size=n_min)
Prompt = Prompt.get_subset(size=n_min )
Non_Prompt = Non_Prompt.get_subset(size=n_min )

print(f"Length of Prompt: {len(Prompt)}")
print(f"Length of Non_Prompt: {len(Non_Prompt)}")
print(f"Length of Background: {len(Background)}")

output_file = os.path.join(model_dir, "train_testr_data.pkl")
pickle.dump(train_test_data, open(output_file, "wb"))
