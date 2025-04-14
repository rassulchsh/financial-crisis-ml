import os
import pandas as pd
from data_pipeline.prepareData import Data
from doExperiment import Experiment

# Create a folder for outputs
os.makedirs("visualizations", exist_ok=True)

# 1. Define Indicator Sets
iv_macro = ["rconsbarro", "iy", "money", "xrusd", "cpi", "ca"]
iv_credit = ["tloans", "debtServ", "yieldCurve", "ltd", "debtgdp",
             "globaltloans", "globalyieldCurve"]
iv_ca = iv_credit + ["hpnom"]
iv_all = iv_macro + iv_ca

# 2. Construct Datasets
print("Preparing datasets...")
df_macro = Data(indicators=iv_macro,
                crisisData="MacroHistory").getReady("Macro")
df_credit = Data(indicators=iv_credit,
                 crisisData="MacroHistory").getReady("Credit")
df_ca = Data(indicators=iv_ca, crisisData="MacroHistory").getReady(
    "Credit & Asset")
df_all = Data(indicators=iv_all, crisisData="MacroHistory").getReady("All")

# 3. Specify Models
models = ["Logit", "KNeighbors", "RandomForest",
          "ExtraTrees", "SVM", "NeuralNet"]

# 4. Create Experiments
print("Creating experiments...")
ex_macroIS = Experiment(df_macro, models, "InSample")
ex_creditIS = Experiment(df_credit, models, "InSample")
ex_caIS = Experiment(df_ca, models, "InSample")
ex_allIS = Experiment(df_all, models, "InSample")

ex_macro = Experiment(df_macro, models, "CrossVal")
ex_credit = Experiment(df_credit, models, "CrossVal")
ex_ca = Experiment(df_ca, models, "CrossVal")
ex_all = Experiment(df_all, models, "CrossVal")

# 5. Run In-Sample Experiments
print("Running In-Sample experiments...")
ex_macroIS.run(disableTqdm=True)
ex_creditIS.run(disableTqdm=True)
ex_caIS.run(disableTqdm=True)
ex_allIS.run(disableTqdm=True)

resIS = pd.concat([ex_macroIS.auc, ex_creditIS.auc,
                  ex_caIS.auc, ex_allIS.auc], axis=0)
resIS.to_csv("visualizations/in_sample_auc.csv", index=False)
print("\n✅ In-Sample AUCs saved to visualizations/in_sample_auc.csv")

# 6. Run Cross-Validation (Out-of-Sample)
print("\nRunning Cross-Validation (100 iterations)...")
n = 100
ex_macro.run(n)
ex_credit.run(n)
ex_ca.run(n)
ex_all.run(n)

resCrossVal = pd.concat(
    [ex_macro.auc, ex_credit.auc, ex_ca.auc, ex_all.auc], axis=0)
resCrossVal.to_csv("visualizations/crossval_auc.csv", index=False)
print("Cross-Validation AUCs saved to visualizations/crossval_auc.csv")

# 7. ROC Plots
print("\nGenerating ROC graphs...")
ex_all.rocGraph()
ex_allIS.rocGraph()
print("ROC plots generated (saved inside experiment logic)")

# 8. Explainability
print("\nExplainability: Correlation Matrix, VIF, and Logit Coefficients")
df_all.correlationMatrix().to_csv(
    "visualizations/correlation_matrix.csv", index=False)
df_all.vif().to_csv("visualizations/vif.csv", index=False)
ex_all.logitCoef().to_csv("visualizations/logit_coefficients.csv")
print("Correlation matrix, VIF, and Logit coefficients saved.")

# 9. ALE Plots
print("\nGenerating ALE plots...")
ex_allIS.ALE(["Logit", "RandomForest", "ExtraTrees"], range(0, 14))
print("ALE plots generated")

# 10. Robustness Checks – ESRB
print("\nRunning Robustness Check: ESRB Crisis Data")
df_esrb = Data(indicators=iv_all, crisisData="ESRB").getReady("ESRB")
ex_esrb = Experiment(df_esrb, models, "CrossVal")
ex_esrb.run(n=100)
ex_esrb.auc.to_csv("visualizations/esrb_auc.csv", index=False)
print("ESRB AUCs saved")

# 11. Robustness Checks – Laeven & Valencia
print("\nRunning Robustness Check: Laeven & Valencia Crisis Data")
df_lv = Data(indicators=iv_all, crisisData="LaevenValencia").getReady(
    "LaevenValencia")
ex_lv = Experiment(df_lv, models, "CrossVal")
ex_lv.run(n=100)
ex_lv.auc.to_csv("visualizations/laevenvalencia_auc.csv", index=False)
print("Laeven & Valencia AUCs saved")

print("\n🎉 All outputs saved in /visualizations folder")
