import os
import pandas as pd
from data_pipeline.prepareData import Data
from doExperiment import Experiment

viz_folder = "visualization"
os.makedirs(viz_folder, exist_ok=True)

output_csv = os.path.join(viz_folder, "forecast_auc_results.csv")

iv_macro = ["rconsbarro", "iy", "money", "xrusd", "cpi", "ca"]
iv_financial = [
    "tloans", "debtServ", "yieldCurve", "ltd", "debtgdp",
    "globaltloans", "globalyieldCurve", "hpnom"
]
iv_all = iv_macro + iv_financial
df_macro = Data(indicators=iv_macro,
                crisisData="MacroHistory").getReady("Macro")
df_financial = Data(indicators=iv_financial,
                    crisisData="MacroHistory").getReady("Financial")
df_all = Data(indicators=iv_all, crisisData="MacroHistory").getReady("All")

models = ["Logit", "KNeighbors", "RandomForest",
          "ExtraTrees", "SVM", "NeuralNet"]

ex_macro = Experiment(df_macro, models, "Forecast")
ex_financial = Experiment(df_financial, models, "Forecast")
ex_all = Experiment(df_all, models, "Forecast")

ex_macro.run(disableTqdm=True)
ex_financial.run(disableTqdm=True)
ex_all.run(disableTqdm=True)

res_auc = pd.concat([ex_macro.auc, ex_financial.auc, ex_all.auc])
res_auc.to_csv(output_csv, index=False)

print("Forecast AUC results saved to:", output_csv)
print(res_auc)
