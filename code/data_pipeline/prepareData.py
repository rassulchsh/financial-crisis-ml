
import pandas as pd
import numpy as np
import data_pipeline.altCrisisData as ac
import seaborn as sn
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


class Data:
    def __init__(self, indicators=[], folder=None, crisisData="MacroHistory", predHor=2, postCrisis=4, diffHor=5, delWW=True):
        if folder == None:
            self.df = pd.read_excel(
                '/Users/rassulraketa/Downloads/JSTdatasetR6.xlsx')
        else:
            self.df = pd.read_excel(folder + r"\JSTdatasetR6.xlsx")

        self.depVar = "crisisRisk"
        self.crisisData = crisisData
        self.indicators = indicators
        self.predHor = predHor
        self.varlist = []
        self.transformedVars = []
        self.countries = []
        self.years = []
        self.len = None
        self.crisisCount = None
        self.isReady = False

        self.varNames = {
            "rconsbarro": "Consumption",
            "iy": "Investment",
            "money": "Broad Money",
            "xrusd": "USD Exchange Rate",
            "cpi": "Consumer Price Index",
            "ca": "Current Account",
            "tloans": "Loans (local)",
            "debtServ": "Debt Service Ratio",
            "yieldCurve": "Yield Curve (local)",
            "ltd": "Banks LDR",
            "debtgdp": "Public Debt",
            "globaltloans": "Loans (global)",
            "globalyieldCurve": "Yield Curve (global)",
            "hpnom": "House Prices"
        }

        if type(self.indicators) != list:
            self.indicators = [self.indicators]
        self.varlist = ["iso", "year", "crisisID"] + \
            [self.depVar] + self.indicators

        #  Indicator Variables
        self.df["yieldCurve"] = self.df["ltrate"] - self.df["stir"]
        self.df["debtServ"] = self.df["tloans"] * self.df["ltrate"] / 100
        self.df["riskInsent"] = self.df["risky_tr"] - self.df["safe_tr"]
        self.transformedVars.append("yieldCurve")

        #  GDP Ratios
        for var in ["tloans", "ca", "money", "debtServ"]:
            self.df[var + "_gdp"] = self.df[var] / self.df["gdp"]

        #  GDP Ratio Change
        for var in ["tloans_gdp", "ca_gdp", "iy", "money_gdp", "debtServ_gdp", "debtgdp"]:
            newName = var + "_ratioDiff" + str(diffHor)
            self.df[newName] = self.df.groupby("iso", group_keys=False)[
                var].diff(diffHor)
            self.transformedVars.append(newName)

        #  Percantage Change
        def lag_pct_change(x):
            lag = np.array(pd.Series(x).shift(diffHor))
            return (x - lag) / lag

        for var in ["cpi", "rconsbarro", "gdp", "unemp", "xrusd", "ltd", "hpnom", "wage"]:
            newName = var + "_pctDiff" + str(diffHor)
            self.df[newName] = self.df.groupby("iso", group_keys=False)[
                var].apply(lag_pct_change)
            self.transformedVars.append(newName)

        #  Global Indicators
        def makeGlobal(varname):
            for year in self.df["year"].unique():
                ix = self.df["year"] == year
                for country in self.df["iso"].unique():
                    perc_pos = self.df.loc[ix.values & (
                        self.df.iso != country).values, varname].mean()
                    if not np.isnan(perc_pos):
                        self.df.loc[ix.values & (
                            self.df.iso == country).values, "global" + varname] = perc_pos
            self.transformedVars.append("global" + varname)

        makeGlobal("tloans_gdp_ratioDiff" + str(diffHor))
        makeGlobal("yieldCurve")

        # Without WWII
        if delWW:
            self.df = self.df[~self.df.year.isin(
                list(range(1914, 1919)) + list(range(1934, 1946)))]

        # Crisis info
        crisisVarNames = {"MacroHistory": "crisisJST", "MacroHistory_old": "crisisJST_old", "LaevenValencia": "crisis_banking",
                          "ESRB": "crisis_esrb"}
        if self.crisisData not in crisisVarNames:
            raise Exception("Invalid Crisis Data specified.")

        elif self.crisisData == "LaevenValencia":
            altCrisis = ac.getLaevenValencia(folder)
            self.df = pd.merge(self.df, altCrisis, how="left")

        elif self.crisisData == "ESRB":
            altCrisis = ac.getESRB(folder)
            self.df = pd.merge(self.df, altCrisis, how="left")

        self.df = self.df.rename(
            columns={crisisVarNames[self.crisisData]: "crisis"})

        self.df[self.depVar] = 0
        self.df["remove"] = 0
        self.df["crisisID"] = 0
        currentID = 1

        for obs in self.df.iterrows():
            year = obs[1][0]
            country = obs[1][2]
            crisis = obs[1][self.df.columns.get_loc("crisis")]
            if crisis == 1:
                for i in range(1, predHor + 1):
                    self.df.loc[(self.df.iso == country) & (
                        self.df.year == year - i), self.depVar] = 1
                    self.df.loc[(self.df.iso == country) & (
                        self.df.year == year - i), "crisisID"] = currentID
                for j in range(0, postCrisis + 1):
                    self.df.loc[(self.df.iso == country) & (
                        self.df.year == year + j), "remove"] = 1
                currentID += 1

        self.df = self.df[["year", "iso", "crisis", "crisisID",
                           "remove"] + [self.depVar] + self.transformedVars]
        nameChange = {}
        for var in self.df.columns:
            nameChange[var] = var.split("_", 1)[0]
        self.df = self.df.rename(columns=nameChange)

        unknownIndicators = np.setdiff1d(self.indicators, self.df.columns)
        if unknownIndicators.size != 0:
            raise Exception(f"Can't find Indicators: {unknownIndicators}")

        self.df = self.df[["year", "iso", "crisis", "crisisID",
                           "remove"] + [self.depVar] + self.indicators]
        self.reloadParameters()
        self.crisisCount = int(self.df.dropna()["crisis"].sum())

    def reloadParameters(self):
        self.df = self.df.sort_values(
            by=["year", "iso"]).reset_index(drop=True)
        self.len = len(self.df)
        self.countries = self.df.iso.unique()
        self.years = self.df.year.unique()

    def getObs(self, country=None, year=None):
        if type(country) == str:
            country = [country]
        if type(year) == int:
            year = [year]
        if country == None:
            country = self.countries
        if year == None:
            year = self.years
        return self.df[(self.df.iso.isin(country)) & (self.df.year.isin(year))]

    def getData(self, ix):
        return self.df.loc[ix]

    def standardize(self):
        df = self.df.copy()
        for indicator in self.indicators:
            mean = self.df[indicator].mean()
            std = self.df[indicator].std()
            df[indicator] = (self.df[indicator] - mean) / std
        return df

    def correlationMatrix(self):
        corr = self.df[self.indicators].corr(method="pearson").abs()
        sn.set(rc={'figure.figsize': (11.5, 8)})
        sn.heatmap(corr, annot=True, cmap="flare")

    def vif(self):
        vif_df = pd.DataFrame()
        vif_df['variable'] = self.indicators
        vif_df['VIF'] = [variance_inflation_factor(
            self.df[self.indicators], i) for i in range(len(self.indicators))]
        return vif_df.sort_values("VIF", ascending=False)

    def getReady(self, name):
        self.df = self.df[self.df.remove == 0].dropna()
        self.df = self.df[self.varlist]
        self.df = self.df[self.df.year <= self.years[-1] - self.predHor]
        self.name = name
        self.reloadParameters()

        self.isReady = True
        print(f"{name}: The final dataset contains {self.len} observations with {self.crisisCount} distinct crisis events.")
        return self
