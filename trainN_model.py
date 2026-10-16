
# training of the model and feature selection process
import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from hipe4ml.model_handler import ModelHandler
import matplotlib.pyplot as plt
from hipe4ml import plot_utils
import ROOT
import shap
from data import Background, Prompt, Non_Prompt
from configN_analysis import *


def add_thesis_info(fig):
    fig.text(0.15, 0.98, r"$\it{This\ Thesis}$", fontsize=14)
    fig.text(0.15, 0.95, r"$1 < p_{\mathrm{T}} < 4\ \mathrm{GeV}/c$", fontsize=14)
    fig.text(0.15, 0.92, r"$1 < \mathit{ct} < 30\ \mathrm{cm}$", fontsize=14)


# configurations
train_test_file = os.path.join(model_dir, "train_test_data.pkl")
model_file = os.path.join(model_dir, "model.pkl")
study_file = os.path.join(model_dir, "optuna_study.pkl")
feature_file = os.path.join(model_dir, "selected_features.txt")
params = {"n_estimators": (0, 500),"max_depth": (2, 5),"learning_rate": (0.01, 0.5),"gamma": (0.5, 2),"colsample_bytree": (0.5, 1.0),"subsample": (0.5, 1.0),"min_child_weight": (1, 10)}

# load data
print("Loading data...")
with open(train_test_file, "rb") as f:
    train_test_data = pickle.load(f)
x_train, y_train, x_test, y_test = train_test_data

# plotting variable distributions
features_for_train = ['DcaPosPV', 'DcaNegPV', 'DcaV0Tracks', 'CosPA', 'DcaV0PV']

vars_to_draw = ['DcaPosPV','DcaNegPV','DcaV0Tracks','CosPA','DcaV0PV']
colors = ['blue', 'red', 'green']
leg_labels = ['Background', 'Prompt', 'Non-Prompt']

x_labels = {
    'DcaPosPV': r'$\mathrm{DCA}_{\mathrm{Pos,PV}}$ (cm)',  
    'DcaNegPV': r'$\mathrm{DCA}_{\mathrm{Neg,PV}}$ (cm)',
    'DcaV0Tracks': r'$\mathrm{DCA}_{\mathrm{V^0,Tracks}}$ (cm)',
    'CosPA': r'$\cos\theta_p$',
    'DcaV0PV': r'$\mathrm{DCA}_{\mathrm{V^0,PV}}$ (cm)', 
    'Mass': r'$Mass$ (GeV/$c^2$)'}

plt.subplots_adjust(left=0.06,bottom=0.06,right=1.5,top=0.96,hspace=0.55,wspace=0.55)
out = plot_utils.plot_distr([Background, Prompt, Non_Prompt],vars_to_draw,colors=colors,labels=leg_labels,log=True,density=True,bins=100,figsize=(25, 13),alpha=0.4,grid=False)

for i, ax in enumerate(out):
    ax.set_title("")
    ax.set_xlabel(x_labels[vars_to_draw[i]], fontsize=16)
    ax.xaxis.set_label_coords(1.0, -0.08)
    ax.set_ylabel('Normalized count', fontsize=16)
    fig = ax.get_figure()
    add_thesis_info(fig)
    fig.savefig(os.path.join(plot_dir, f"distribution_{i}.png"),dpi=300,bbox_inches="tight")
plt.show()
print("running feature selection...")

x_labels1 = {
    'DcaPosPV': r'$\mathrm{DCA}_{\mathrm{Pos,PV}}$',  
    'DcaNegPV': r'$\mathrm{DCA}_{\mathrm{Neg,PV}}$',
    'DcaV0Tracks': r'$\mathrm{DCA}_{\mathrm{V^0,Tracks}}$',
    'CosPA': r'$\cos\theta_p$',
    'DcaV0PV': r'$\mathrm{DCA}_{\mathrm{V^0,PV}}$', 
    'Mass': r'$Mass$ (GeV/$c^2$)'}

Background_plot = Background[vars_to_draw].rename(columns=x_labels1)
Prompt_plot = Prompt[vars_to_draw].rename(columns=x_labels1)
Non_Prompt_plot = Non_Prompt[vars_to_draw].rename(columns=x_labels1)
display_vars = [x_labels1[var] for var in vars_to_draw]

out1 = plot_utils.plot_corr([Background_plot, Prompt_plot, Non_Prompt_plot],display_vars,leg_labels)
for i, fig in enumerate(out1):
    add_thesis_info(fig)
    fig.savefig(os.path.join(plot_dir, f"correlation_matrix_{i}.png"),dpi=300,bbox_inches="tight")
plt.show()

# definition of the training features
# Summary
print("\n##################################")
print("Selected variables:")
print("####################################\n")
print(features_for_train)

# build ModelHandler
model_clf = xgb.XGBClassifier(tree_method='hist', n_jobs= 7, enable_categorical=True, objective = 'multi:softprob', num_class = 3, eval_metric="mlogloss")
model_hdl = ModelHandler(model_clf, features_for_train)

# save training columns files
with open(os.path.join(model_dir, "training_columns.txt"), "w") as f:
    for col in model_hdl.get_training_columns():
        f.write(col + "\n")
print("Training columns saved in training_columns.txt")


best_params = model_hdl.get_model_params()

# hyperparameter optimization Optuna
if os.path.exists(study_file):
    print("Loading existing Optuna study...")
    with open(study_file, "rb") as f:
        study = pickle.load(f)
    best_params = study.best_params
    print(" Best params:", best_params)
    # rebuild model cleanly
    model_clf = xgb.XGBClassifier(tree_method='hist',n_jobs= 7,enable_categorical=True,**best_params, objective = 'multi:softprob',num_class = 3, eval_metric="mlogloss")
    model_hdl = ModelHandler(model_clf, features_for_train)
else:
    print("Running Optuna optimization...")
    study = model_hdl.optimize_params_optuna(train_test_data,params,cross_val_scoring="roc_auc_ovr",nfold=5,n_jobs= 7,n_trials=100,direction="maximize",save_study=study_file)

# final training
model_hdl.train_test_model(train_test_data, multi_class_opt="ovr")

# save trained model
print("Saving model ...")
model_hdl.dump_model_handler(model_file)

# save hyper-parameters
with open(os.path.join(model_dir, "model_parameters.txt"), "w") as f:
    for k, v in model_hdl.get_model_params().items():
        f.write(f"{k}: {v}\n")
print("Parameters saved in model_parameters.txt")

# feature importance plotting
def plot_feature_imp(train_test_data, model_hdl, labels, plot_dir,n_sample=100000, approximate=True):
    x_train = train_test_data[0]
    y_train = train_test_data[1].ravel()
    x_test = train_test_data[2]
    y_test = train_test_data[3].ravel()

    class_labels, class_counts = np.unique(y_train, return_counts=True)
    n_classes = len(class_labels)

    for class_count in class_counts:
        n_sample = min(n_sample, class_count)

    subs = []
    for class_lab in class_labels:
        subs.append(x_train[y_train == class_lab].sample(n_sample))

    df_subs = pd.concat(subs)
    df_subs = df_subs[model_hdl.get_training_columns()]

    explainer = shap.TreeExplainer(model_hdl.get_original_model())
    shap_values = explainer.shap_values(df_subs,approximate=approximate)
    df_subs_display = df_subs.rename(columns=x_labels1)

    res = []
    if n_classes <= 2:
        fig = plt.figure(figsize=(18, 9))
        shap.summary_plot(shap_values,df_subs_display,plot_size=(18, 9),class_names=labels,show=False)
        add_thesis_info(fig)
        res.append(fig)
        fig = plt.figure(figsize=(18, 9))

        shap.summary_plot(shap_values,df_subs_display,plot_type='bar',plot_size=(18, 9),class_names=labels,show=False)
        add_thesis_info(fig)
        res.append(fig)
    else:
        shap_values_transposed = shap_values.transpose(2, 0, 1)
        for i_class in range(n_classes):
            fig = plt.figure(figsize=(18, 9))
            shap.summary_plot(shap_values_transposed[i_class],df_subs_display,plot_size=(18, 9),class_names=labels,show=False)
            add_thesis_info(fig)
            res.append(fig)
        fig = plt.figure(figsize=(18, 9))
        shap.summary_plot(list(shap_values_transposed),df_subs_display,plot_type='bar',plot_size=(18, 9),class_names=labels,show=False)
        add_thesis_info(fig)
        res.append(fig)

    # Save SHAP figures
    for i, fig in enumerate(res):
        fig.savefig(os.path.join(plot_dir,f"shap_feature_importance_{i}.png"),dpi=300,bbox_inches="tight")

    return res

model_hdl = ModelHandler()
model_hdl.load_model_handler(model_file)
res = plot_feature_imp(train_test_data, model_hdl, leg_labels,plot_dir, n_sample=100000, approximate=True)



# prediction of the model
y_pred_train = model_hdl.predict(x_train)
y_pred_test = model_hdl.predict(x_test)

# plotting
plt.rcParams["figure.figsize"] = (10, 7)
ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 60,output_margin=False, labels=leg_labels,logscale=True, density=True )
plt.show()
#roc_train_test_fig = plot_utils.plot_roc_train_test(y_train, y_pred_train,y_test,y_pred_test,leg_labels, multi_class_opt="ovo")
roc_train_test_fig = plot_utils.plot_roc_train_test(y_test,y_pred_test, y_train, y_pred_train,leg_labels, multi_class_opt="ovr")

# save BDT plots as PNG images
for i, fig in enumerate(ml_out_fig):
    add_thesis_info(fig)
    fig.savefig(os.path.join(plot_dir, f"bdt_output_train_test_{i}.png"),dpi=300,bbox_inches="tight")

# save ROC curve as PNG image
add_thesis_info(roc_train_test_fig)
roc_train_test_fig.savefig(os.path.join(plot_dir, "roc_curve.png"),dpi=300,bbox_inches="tight")

# ROOT canvas for train/test output
for i in range(len(ml_out_fig)):    
    c_bdt = ROOT.TCanvas(f"c_bdt_{i}", f"BDT output {i}", 900, 700)
    img_bdt = ROOT.TImage.Open(os.path.join(plot_dir, f"bdt_output_train_test_{i}.png"))
    img_bdt.Draw()
    c_bdt.Update()
    c_bdt.SaveAs(os.path.join(plot_dir, f"bdt_output_train_test_root_{i}.png"))

# ROOT canvas for ROC curve
c_roc = ROOT.TCanvas("c_roc", "ROC curve", 900, 700)

img_roc = ROOT.TImage.Open(os.path.join(plot_dir, "roc_curve.png"))
img_roc.Draw()
c_roc.Update()
c_roc.SaveAs(os.path.join(plot_dir, "roc_curve_root.png"))

# ROOT canvas for correlation matrix
for i in range(len(out1)):
    c_corr = ROOT.TCanvas(f"c_corr_{i}",f"Correlation matrix {i}",900,700)
    img_corr = ROOT.TImage.Open(os.path.join(plot_dir, f"correlation_matrix_{i}.png"))
    img_corr.Draw()
    c_corr.Update()
    c_corr.SaveAs(os.path.join(plot_dir, f"correlation_matrix_root_{i}.png"))

# Root canvas for features importance
outfile = ROOT.TFile(os.path.join(root_dir, "training_plots.root"),"RECREATE")

for i in range(len(res)):
    c_shap = ROOT.TCanvas(f"c_shap_{i}",f"SHAP {i}",900,700)
    img = ROOT.TImage.Open(os.path.join(plot_dir, f"shap_feature_importance_{i}.png"))
    img.Draw()
    c_shap.Update()
    c_shap.SaveAs(os.path.join(plot_dir, f"shap_feature_importance_root_{i}.png"))
    c_shap.Write()

c_bdt.Write()
c_roc.Write()
c_corr.Write()
outfile.Close()

# end
print("training completed")
print("model saved :", model_file)
