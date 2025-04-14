import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import data_pipeline.prepareData as pd
import doExperiment as de

viz_folder = "visualization"
if not os.path.exists(viz_folder):
    os.makedirs(viz_folder)

selected_country = "DEU"
indicators_to_use = [
    'cpi',
    'gdp',
    'debtgdp',
]

data = pd.Data(
    indicators=indicators_to_use,
    predHor=2,
    postCrisis=2,
    diffHor=5
)
data.getReady('CrisisData')

print(
    f"Full dataset contains {data.len} observations from {len(data.countries)} countries.")

exp = de.Experiment(data, ['Logit', 'RandomForest'], 'CrossVal')
exp.run()
exp.rocGraph(save=True)
print("ROC graph saved in the visualization folder.")

rf_params = None
for model_name, params in exp.searchRes:
    if model_name == "RandomForest":
        rf_params = params
        break

if rf_params is None:
    print("RandomForest model not found in search results; using default parameters.")
    rf_params = {}

scaler = StandardScaler()
X_full = scaler.fit_transform(data.df[data.indicators])
y_full = data.df[data.depVar]

best_rf = RandomForestClassifier(
    random_state=1, n_estimators=1000, **rf_params)
best_rf.fit(X_full, y_full)

pred_probs_full = best_rf.predict_proba(X_full)[:, 1]
data.df['predicted_prob'] = pred_probs_full

country_data = data.getObs(country=selected_country)
if country_data.empty:
    sys.exit(f"No data found for country: {selected_country}")
else:
    print(f"Data for {selected_country} has {len(country_data)} observations.")

country_data = data.df[data.df['iso'] ==
                       selected_country].reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.plot(country_data['year'], country_data['predicted_prob'],
         marker='o', linestyle='-', label='Predicted Crisis Probability')
plt.xlabel('Year')
plt.ylabel('Predicted Crisis Probability')
plt.title(f'Crisis Prediction for {selected_country}')
plt.legend()
plt.grid(True)

out_path = os.path.join(viz_folder, f"predictions_{selected_country}.png")
plt.savefig(out_path)
plt.show()

print(f"Prediction chart saved in {out_path}")
