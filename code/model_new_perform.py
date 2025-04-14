import os
import pandas as pd
from data_pipeline.prepareData import Data
from new_models_experiment import Experiment

os.makedirs("visualizations_new", exist_ok=True)

iv_macro = ["rconsbarro", "iy", "money", "xrusd", "cpi", "ca"]
iv_credit = ["tloans", "debtServ", "yieldCurve", "ltd", "debtgdp",
             "globaltloans", "globalyieldCurve"]
iv_ca = iv_credit + ["hpnom"]
iv_all = iv_macro + iv_ca

print("Preparing datasets...")
df_macro = Data(indicators=iv_macro,
                crisisData="MacroHistory").getReady("Macro")
df_credit = Data(indicators=iv_credit,
                 crisisData="MacroHistory").getReady("Credit")
df_ca = Data(indicators=iv_ca, crisisData="MacroHistory").getReady(
    "Credit & Asset")
df_all = Data(indicators=iv_all, crisisData="MacroHistory").getReady("All")

models = ["XGBoost", "LSTM"]


print("Creating experiments...")
ex_macroIS = Experiment(df_macro, models, "InSample")
ex_creditIS = Experiment(df_credit, models, "InSample")
ex_caIS = Experiment(df_ca, models, "InSample")
ex_allIS = Experiment(df_all, models, "InSample")

ex_macro = Experiment(df_macro, models, "CrossVal")
ex_credit = Experiment(df_credit, models, "CrossVal")
ex_ca = Experiment(df_ca, models, "CrossVal")
ex_all = Experiment(df_all, models, "CrossVal")

print("Running In-Sample experiments...")
ex_macroIS.run(disableTqdm=True)
ex_creditIS.run(disableTqdm=True)
ex_caIS.run(disableTqdm=True)
ex_allIS.run(disableTqdm=True)

resIS = pd.concat([ex_macroIS.auc, ex_creditIS.auc,
                   ex_caIS.auc, ex_allIS.auc], axis=0)
resIS.to_csv("visualizations/in_sample_auc.csv", index=False)
print("\nIn-Sample AUCs saved to visualizations_new/in_sample_auc.csv")

print("\nRunning Cross-Validation (100 iterations)...")
n = 100
ex_macro.run(n)
ex_credit.run(n)
ex_ca.run(n)
ex_all.run(n)

resCrossVal = pd.concat([ex_macro.auc, ex_credit.auc,
                         ex_ca.auc, ex_all.auc], axis=0)
resCrossVal.to_csv("visualizations/crossval_auc.csv", index=False)
print("Cross-Validation AUCs saved to visualizations_new/crossval_auc.csv")

print("\nGenerating ROC graphs...")
ex_all.rocGraph()
ex_allIS.rocGraph()
print("ROC plots generated (saved inside experiment logic)")
print("\nGenerating ALE plots for XGBoost and LSTM...")
ex_allIS.ALE(models, range(0, len(iv_all)))
print("ALE plots generated")
print("\nRunning Robustness Check: ESRB Crisis Data")
df_esrb = Data(indicators=iv_all, crisisData="ESRB").getReady("ESRB")
ex_esrb = Experiment(df_esrb, models, "CrossVal")
ex_esrb.run(n=100)
ex_esrb.auc.to_csv("visualizations_new/esrb_auc.csv", index=False)
print("ESRB AUCs saved")
print("\nRunning Robustness Check: Laeven & Valencia Crisis Data")
df_lv = Data(indicators=iv_all, crisisData="LaevenValencia").getReady(
    "LaevenValencia")
ex_lv = Experiment(df_lv, models, "CrossVal")
ex_lv.run(n=100)
ex_lv.auc.to_csv("visualizations_new/laevenvalencia_auc.csv", index=False)
print("Laeven & Valencia AUCs saved")

print("All outputs saved in /visualizations_new folder")
