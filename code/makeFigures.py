import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_pipeline.prepareData import Data


import os
os.makedirs("visualization", exist_ok=True)

# --- 1. MacroHistory Crisis Events ---
print("Plotting MacroHistory crisis events...")
d1 = Data()
crisisYears = d1.df[d1.df.crisis == 1][["year", "iso"]
                                       ].sort_values("iso").reset_index(drop=True)

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_theme(style="whitegrid", rc={'savefig.dpi': 300})
sns.stripplot(data=crisisYears, x="year", y="iso", size=8, jitter=False,
              palette=sns.color_palette("husl", 18), hue="iso", legend=None)
plt.grid(axis="y")
plt.xlabel("Year")
plt.ylabel("Country")
plt.xticks(fontsize=12)
plt.xticks(list(range(1870, 2021, 10)))
plt.tight_layout()
plt.savefig("visualization/crisisYears_macro.png", bbox_inches='tight')
plt.clf()

# --- 2. Laeven & Valencia Crisis Events ---
print("Plotting Laeven & Valencia crisis events...")
d2 = Data(crisisData="LaevenValencia")
crisisYears = d2.df[d2.df.crisis == 1][["year", "iso"]
                                       ].sort_values("iso").reset_index(drop=True)

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_theme(style="whitegrid")
sns.stripplot(data=crisisYears, x="year", y="iso", size=8, jitter=False,
              palette=sns.color_palette("husl", 16), hue="iso", legend=None)
plt.grid(axis="y")
plt.xlabel("Year")
plt.ylabel("Country")
plt.xticks(fontsize=12, rotation=45)
plt.xticks(list(range(1970, 2021, 4)))
plt.tight_layout()
plt.savefig("visualization/crisisYears_LV.png", bbox_inches='tight')
plt.clf()

# --- 3. ESRB Crisis Events ---
print("Plotting ESRB crisis events...")
d3 = Data(crisisData="ESRB")
crisisYears = d3.df[d3.df.crisis == 1][["year", "iso"]
                                       ].sort_values("iso").reset_index(drop=True)

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_theme(style="whitegrid")
sns.stripplot(data=crisisYears, x="year", y="iso", size=8, jitter=False,
              palette=sns.color_palette("husl", 13), hue="iso", legend=None)
plt.grid(axis="y")
plt.xlabel("Year")
plt.ylabel("Country")
plt.xticks(fontsize=12, rotation=45)
plt.xticks(list(range(1970, 2021, 4)))
plt.tight_layout()
plt.savefig("visualization/crisisYears_ESRB.png", bbox_inches='tight')
plt.clf()

print("✅ All crisis figures saved to the 'visualization' folder.")
