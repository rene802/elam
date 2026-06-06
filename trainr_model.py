
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
from load_datar import Background, Prompt, Non_Prompt
from config import model_dir, plot_dir, root_dir, log_dir


# configuration
train_test_file = os.path.join(model_dir, "train_testp_data.pkl")
model_file = os.path.join(model_dir, "model_p.pkl")
study_file = os.path.join(model_dir, "optuna_studyp.pkl")
feature_file = os.path.join(model_dir, "selected_features.txt")
params = {"n_estimators": (0, 200),"max_depth": (2, 5),"learning_rate": (0.01, 0.09),"gamma": (1e-3, 1),"colsample_bytree": (0.5, 1.0),"subsample": (0.5, 1.0),"min_child_weight": (1, 2)}
importance_threshold = 0.97
correlation_threshold = 0.80

# load data
print("Loading data...")
with open(train_test_file, "rb") as f:
    train_test_data = pickle.load(f)
x_train, y_train, x_test, y_test = train_test_data

# plotting variable distributions
vars_to_draw = list(set(Background.get_var_names()) & set(Prompt.get_var_names()) & set(Non_Prompt.get_var_names()))
colors = ['blue', 'red', 'green']
leg_labels = ['Background', 'Prompt', 'Non-Prompt']

plt.subplots_adjust(left=0.06, bottom=0.06, right=1.5, top=0.96, hspace=0.55, wspace=0.55)
out = plot_utils.plot_distr([Background, Prompt, Non_Prompt], vars_to_draw, colors=colors, labels=leg_labels, log=True, density=True, bins=100, figsize=(25, 13), alpha=0.4, grid=False)
plt.show()

out1 = plot_utils.plot_corr([Background, Prompt, Non_Prompt], vars_to_draw, leg_labels)
for i, fig in enumerate(out1):
    fig.savefig(os.path.join(plot_dir, f"correlation_matrix_{i}.png"),dpi=300)
plt.show()


# feature selection
print("running feature selection...")

# building dataframe
df_bkg = Background.get_data_frame().copy()
df_bkg["label"] = 0
df_prompt = Prompt.get_data_frame().copy()
df_prompt["label"] = 1
df_nonprompt = Non_Prompt.get_data_frame().copy()
df_nonprompt["label"] = 2
df_train = pd.concat([df_bkg, df_prompt, df_nonprompt], ignore_index=True)

# variables candidates
variables_to_exclude = ['fMass', 'fQtAP', 'fTpcNsigmaPos','fTpcNsigmaNeg','fPt','fEta'] #, 'fTpcNsigmaPos', 'fTpcNsigmaNeg', 'fPt',,'fRadius','fEta']
features_candidates = [v for v in vars_to_draw if v not in variables_to_exclude]
print("\ncandidates variables:\n", features_candidates)

# remove NaN and inf values
x = df_train[features_candidates].copy()
x.replace([np.inf, -np.inf], np.nan, inplace=True)
mask = x.notna().all(axis=1)
x = x[mask]
y = df_train.loc[mask, "label"]

# temporary XGBoost training
temp_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3,n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42,gamma=0.1, min_child_weight=1,eval_metric='mlogloss')
temp_model.fit(x, y)

# feature importance
importance = pd.Series(temp_model.feature_importances_, index=features_candidates)
importance = importance.sort_values(ascending=False)
print("\nfeature importance:\n", importance)

# keep variables contributing of the importance
cum_importance = (importance.cumsum() / importance.sum())
features_selection = []
for feature, imp in cum_importance.items():
    features_selection.append(feature)
    if imp >= importance_threshold:
        break
print("\nvariables retained after high cumulated importance:\n", features_selection)
for feature in features_selection:
    print(f"{feature}: {importance[feature]:.4f} ({cum_importance[feature]*100:.2f}%)")

# correlation matrix check    
correlation_matrix = df_train[features_selection].corr().abs()

# remove highly correlated variables
selected_features = []
removed_features = set()
ordered_features = [feature for feature in importance.index if feature in features_selection]
for feature in ordered_features:
    if feature in removed_features:
        continue
    selected_features.append(feature)
    for other_feature in ordered_features:
        if other_feature == feature:
            continue
        if other_feature in removed_features:
            continue
        corr_value = correlation_matrix.loc[feature,other_feature]
        if corr_value > correlation_threshold:
            removed_features.add(other_feature)
            print(f"remove{other_feature}"
                  f"(corr={corr_value:.3f})"
                  f"with {feature}")

# final feature list
features_for_train = selected_features
print("final features for training")   
for feature in features_for_train:
    print (feature)

# final training for the selection of features_for_train
x_final = df_train[features_for_train].copy()
x_final = x_final.replace([np.inf, -np.inf], np.nan)
mask_final = x_final.notna().all(axis=1)
x_final = x_final[mask_final]
y_final = df_train.loc[mask_final, "label"]    
final_model = xgb.XGBClassifier(objective="multi:softprob",
    num_class=3,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    gamma=0.1,
    min_child_weight=1,
    eval_metric="mlogloss")
final_model.fit(x_final, y_final)

# Summary
print("\n====================================")
print("SUMMARY")
print("====================================")
print(f"Initial variables       : {len(features_candidates)}")
print(f"After importance   : {len(features_selection)}")
print(f"After correlation cut   : {len(features_for_train)}")

print("\nSelected variables:")
print(features_for_train)

# build ModelHandler
model_clf = xgb.XGBClassifier(tree_method='hist', enable_categorical=True)
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
    model_clf = xgb.XGBClassifier(
        tree_method='hist',
        enable_categorical=True,
        **best_params
    )
    model_hdl = ModelHandler(model_clf, features_for_train)
else:
    print("Running Optuna optimization...")
    study = model_hdl.optimize_params_optuna(
        train_test_data,
        params,
        cross_val_scoring="roc_auc_ovo",
        nfold=5,
        n_jobs= -1,
        n_trials=200,
        direction="maximize",
        save_study=study_file
    )

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
def plot_feature_imp(train_test_data, model_hdl, labels,plot_dir, n_sample=100000, approximate=True):
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
        shap_values = explainer.shap_values(df_subs, approximate=approximate)
        res = []

        if n_classes <= 2:
            res.append(plt.figure(figsize=(18, 9)))
            shap.summary_plot(shap_values, df_subs, plot_size=(
                18, 9), class_names=labels, show=False)
            res.append(plt.figure(figsize=(18, 9)))
            shap.summary_plot(shap_values, df_subs, plot_type='bar', plot_size=(
                18, 9), class_names=labels, show=False)
        else:
            shap_values_transposed = shap_values.transpose(2, 0, 1)
            for i_class in range(n_classes):
                res.append(plt.figure(figsize=(18, 9)))
                shap.summary_plot(shap_values_transposed[i_class], df_subs, plot_size=(
                    18, 9), class_names=labels, show=False)
            res.append(plt.figure(figsize=(18, 9)))
            shap.summary_plot(list(shap_values_transposed), df_subs, plot_type='bar', plot_size=(
                18, 9), class_names=labels, show=False)

            for i, fig in enumerate(res):
                fig.savefig(os.path.join(plot_dir, f"shap_feature_importance_{i}.png"),dpi=300,bbox_inches="tight")

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
roc_train_test_fig = plot_utils.plot_roc(train_test_data[3], y_pred_test, None, leg_labels, multi_class_opt="ovo")

# save plotting in png images
for i, fig in enumerate(ml_out_fig):
    fig.savefig(os.path.join(plot_dir, f"bdt_output_train_test_{i}.png"),dpi=300)
    roc_train_test_fig.savefig(os.path.join(plot_dir, "roc_curve.png"),dpi=300)

# ROOT canvas for train/test output
for i in range(len(ml_out_fig)):    
    c_bdt = ROOT.TCanvas(f"c_bdt_{i}", f"BDT output {i}", 900, 700)
    
    img_bdt = ROOT.TImage.Open(os.path.join(plot_dir, f"bdt_output_train_test_{i}.png"))
    img_bdt.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC() 
    latex.SetTextSize(0.02)
    latex.DrawLatex(0.15, 0.85, "#it{This Thesis}")
    latex.DrawLatex(0.15, 0.83, "1 < #it{p}_{T} < 4 GeV/#it{c}")
    
    c_bdt.Update()
    c_bdt.SaveAs(os.path.join(plot_dir, f"bdt_output_train_test_root_{i}.png"))

# ROOT canvas for ROC curve
c_roc = ROOT.TCanvas("c_roc", "ROC curve", 900, 700)

img_roc = ROOT.TImage.Open(os.path.join(plot_dir, "roc_curve.png"))
img_roc.Draw()

latex = ROOT.TLatex()
latex.SetNDC() 
latex.SetTextSize(0.02)
latex.DrawLatex(0.15, 0.85, "#it{This Thesis}")
latex.DrawLatex(0.15, 0.83, "1 < #it{p}_{T} < 4 GeV/#it{c}")

c_roc.Update()
c_roc.SaveAs(os.path.join(plot_dir, "roc_curve_root.png"))

# ROOT canvas for correlation matrix
for i in range(len(out1)):
    c_corr = ROOT.TCanvas(f"c_corr_{i}",f"Correlation matrix {i}",900,700)
    img_corr = ROOT.TImage.Open(os.path.join(plot_dir, f"correlation_matrix_{i}.png"))
    img_corr.Draw()
    latex = ROOT.TLatex()
    latex.SetNDC() 
    latex.SetTextSize(0.02)
    latex.DrawLatex(0.15, 0.85, "#it{This Thesis}")
    latex.DrawLatex(0.15, 0.83, "1 < #it{p}_{T} < 4 GeV/#it{c}")
    c_corr.Update()
    c_corr.SaveAs(os.path.join(plot_dir, f"correlation_matrix_root_{i}.png"))

# Save ROOT objects
outfile = ROOT.TFile(os.path.join(root_dir, "training_plots.root"),"RECREATE")

for i in range(len(res)):
    c_shap = ROOT.TCanvas(f"c_shap_{i}",f"SHAP {i}",900,700)

    img = ROOT.TImage.Open(os.path.join(plot_dir, f"shap_feature_importance_{i}.png"))
    img.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.02)
    latex.DrawLatex(0.15, 0.85, "#it{This Thesis}")
    latex.DrawLatex(0.15, 0.83, "1 < #it{p}_{T} < 4 GeV/#it{c}")

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
