#!/usr/bin/env python
import data_pipeline.prepareData as pds
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import svm as sk_svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from data_pipeline.utils import CustomGroupKFold, CustomTimeSeriesSplit, forecastExp, InSampleSplit as BaseInSampleSplit
import statsmodels.api as sm
from alibi.explainers import ALE, plot_ale
import xgboost as xgb

from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import _estimator_html_repr

# --- PATCH: Modify InSampleSplit to reshape 3D input (if middle dim is 1) into 2D ---


class InSampleSplit(BaseInSampleSplit):
    def split(self, X, y=None, groups=None):
        # If the input is 3D with a singleton in axis 1, reshape it to 2D.
        if X.ndim == 3 and X.shape[1] == 1:
            X = X.reshape(X.shape[0], -1)
        return super().split(X, y, groups)

# Transformer to reshape data for the LSTM model.


class ReshapeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expand dims along axis 1: (n_samples, n_features) -> (n_samples, 1, n_features)
        return np.expand_dims(X, axis=1)


def create_lstm_model(n_features, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, n_features), return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam",
                  loss="binary_crossentropy", metrics=["AUC"])
    return model


class MyKerasClassifier(KerasClassifier):
    def __sklearn_tags__(self):
        return {
            "requires_y": True,
            "non_deterministic": True,
            "preserves_dtype": [],
            "binary_only": False,
        }


class Experiment:
    def __init__(self, data, models, expType, rs=1):
        self.data = data
        self.models = ["Random Assignment"] + models
        self.expType = expType
        validModels = ["Random Assignment", "Logit", "RandomForest",
                       "ExtraTrees", "SVM", "NeuralNet", "KNeighbors", "XGBoost", "LSTM"]
        validExpTypes = ["InSample", "CrossVal", "Forecast", 0, 1, 2]
        unvalidModels = [m for m in self.models if m not in validModels]
        if unvalidModels:
            raise Exception(f"Invalid Model Type specified: {unvalidModels}")
        if self.expType not in validExpTypes:
            raise Exception(
                f"Invalid Experiment Type specified: {self.expType}")

        # Define a proper build function for the LSTM.
        n_features = len(self.data.indicators)

        def build_lstm():
            return create_lstm_model(n_features)

        self.modelClasses = {
            "Random Assignment": DummyClassifier(strategy="constant", constant=0),
            "Logit": LogisticRegression(random_state=rs, max_iter=1000),
            "RandomForest": RandomForestClassifier(random_state=rs, n_estimators=1000),
            "ExtraTrees": ExtraTreesClassifier(random_state=rs, n_estimators=1000),
            "SVM": sk_svm.SVC(random_state=rs, probability=True),
            "NeuralNet": MLPClassifier(random_state=rs, max_iter=10000, alpha=2),
            "KNeighbors": KNeighborsClassifier(n_neighbors=100),
            "XGBoost": xgb.XGBClassifier(random_state=rs, use_label_encoder=False, eval_metric="logloss"),
            "LSTM": MyKerasClassifier(build_fn=build_lstm, epochs=50, batch_size=16, verbose=0)
        }

        # Use our patched InSampleSplit for the "InSample" experiment type.
        iteratorTypes = {
            "InSample": InSampleSplit(),
            "CrossVal": CustomGroupKFold(),
            "Forecast": forecastExp(year=1980)
        }
        self.iterator = iteratorTypes[self.expType]

    def run(self, n=1, disableTqdm=False):
        self.roc = pd.DataFrame()
        self.auc = pd.DataFrame()
        self.searchRes = []
        if self.expType in ["InSample", "Forecast"]:
            n = 1

        for model in self.models:
            pipe = self._buildPipeline(model)
            yTrue = []
            predictions = []
            for _ in tqdm(range(n), desc=f"{self.data.name}: {model}", disable=disableTqdm):
                for ixTrain, ixTest in self.iterator.split(self.data.df):
                    xTrain = self.data.df.loc[ixTrain]
                    yTrain = self.data.df[self.data.depVar].loc[ixTrain]
                    xTest = self.data.df.loc[ixTest]
                    yTest = self.data.df[self.data.depVar].loc[ixTest]

                    pipe.fit(xTrain, yTrain)
                    yTrue = np.append(yTrue, yTest)
                    predictions = np.append(
                        predictions, pipe.predict_proba(xTest)[:, 1])
                    try:
                        best = pipe.named_steps["BestEstimator"].best_params_
                        self.searchRes.append((str(model), best))
                    except Exception:
                        self.searchRes.append((str(model), {}))
            fpr, tpr, threshold = metrics.roc_curve(yTrue, predictions)
            self.roc = pd.concat((
                self.roc,
                pd.DataFrame({"Model": model, "FPR": fpr,
                             "TPR": tpr, "Threshold": threshold})
            ))
            foldAUC = metrics.roc_auc_score(yTrue, predictions)
            self.auc = pd.concat((
                self.auc,
                pd.DataFrame(
                    {"Set": self.data.name, "Model": [model], "AUC": foldAUC})
            ))
        self.auc = self.auc.sort_values(
            "AUC", ascending=False).reset_index(drop=True)
        self.roc = self.roc.reset_index(drop=True)

    def _buildPipeline(self, model):
        preprocess = ColumnTransformer(
            [("Standardize", StandardScaler(), self.data.indicators)],
            verbose_feature_names_out=False
        )
        paras = {
            "Random Assignment": {},
            "Logit": {"penalty": ["l2", "none"]},
            "RandomForest": {"max_depth": [2, 4, 6]},
            "ExtraTrees": {"max_depth": [2, 4, 6]},
            "SVM": {"gamma": ["scale", "auto"], "kernel": ["rbf", "linear", "poly", "sigmoid"]},
            "NeuralNet": {"hidden_layer_sizes": [(8, 8, 8), (20,)], "activation": ["tanh", "relu"]},
            "KNeighbors": {"weights": ["uniform", "distance"], "n_neighbors": [50, 75, 100]},
            "XGBoost": {"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2],
                        "n_estimators": [100, 500, 1000]},
            "LSTM": {"epochs": [50, 100], "batch_size": [16, 32]}
        }
        # For LSTM, add the ReshapeTransformer step to adjust input shape.
        if model == "LSTM":
            steps = [
                ("Preprocess", preprocess),
                ("Reshape", ReshapeTransformer()),
                ("BestEstimator", GridSearchCV(
                    estimator=self.modelClasses[model],
                    param_grid=paras[model],
                    scoring="roc_auc",
                    cv=InSampleSplit(),
                    n_jobs=-1
                ))
            ]
        else:
            steps = [
                ("Preprocess", preprocess),
                ("BestEstimator", GridSearchCV(
                    estimator=self.modelClasses[model],
                    param_grid=paras[model],
                    scoring="roc_auc",
                    cv=InSampleSplit(),
                    n_jobs=-1
                ))
            ]
        return Pipeline(steps)

    def rocGraph(self, save=False):
        sns.set_theme(style="whitegrid", palette="tab10",
                      rc={'savefig.dpi': 300})
        plt.figure(figsize=(6, 6))
        plot = sns.lineplot(data=self.roc, x="FPR", y="TPR",
                            hue="Model", estimator=None, n_boot=0)
        plot.lines[0].set_linestyle("--")
        plot.set(xlabel="False Positive Rate (FPR)",
                 ylabel="True Positive Rate (TPR)")
        labels = []
        for mo in self.models:
            auc_val = self.auc.loc[self.auc["Model"] == mo, "AUC"]
            if len(auc_val) > 0:
                labels.append(f"{mo}: (AUC = {round(auc_val.iloc[0], 3)})")
            else:
                labels.append(mo)
        plt.legend(labels=labels, fontsize="small")
        if save:
            plt.savefig("visualization/roc.png")

    def logitCoef(self):
        df_std = self.data.standardize()
        exogWConst = sm.add_constant(df_std[self.data.indicators])
        mod = sm.Logit(df_std[self.data.depVar], exogWConst)
        fii = mod.fit()
        coef = fii.summary2().tables[1]
        for var in coef.index:
            p_val = coef.loc[var, "P>|z|"]
            if p_val < 0.01:
                print(f"{var} is significant at 1%")
            elif p_val < 0.05:
                print(f"{var} is significant at 5%")
            elif p_val < 0.1:
                print(f"{var} is significant at 10%")
        return coef

    def ALE(self, modelTypes, var):
        df_std = self.data.standardize()
        x = df_std[self.data.indicators].to_numpy()
        y = df_std[self.data.depVar].to_numpy()
        exps = {}
        for modelType in modelTypes:
            try:
                for paras in self.searchRes:
                    if paras[0] == modelType:
                        model = self.modelClasses[modelType].set_params(
                            **paras[1])
                        break
                model.fit(x, y)
            except AttributeError:
                raise ValueError(
                    "No Hyperparameter Search has been conducted yet. Please use .run() first.")
            pred = model.predict_proba(x)[:, 1]
            fpr, tpr, threshold = metrics.roc_curve(y, pred)
            print(f"{modelType} AUC: {metrics.auc(fpr, tpr)}")

            def predict_crisis(x):
                return model.predict_proba(x)[:, 1]
            ale = ALE(predict_crisis, feature_names=list(
                map(self.data.varNames.get, self.data.indicators)))
            exps[modelType] = ale.explain(x, features=var)
        sns.set_theme(style="whitegrid", palette="tab10")
        fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(12, 16))
        for i, model in enumerate(exps.keys()):
            plot_ale(exps[model], n_cols=4, ax=ax, line_kw={"label": model})
        fig.delaxes(ax[4, 2])
        ax[0, 0].get_legend().remove()
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right",
                   bbox_to_anchor=(0.83, 0.145))


if __name__ == "__main__":
    # Create a data object using the 'Data' class from prepareData.
    # Provide indicator names matching the renamed columns.
    myIndicators = ["tloans", "cpi", "gdp"]
    myData = pds.Data(indicators=myIndicators,
                      folder=None,
                      crisisData="MacroHistory",
                      predHor=2,
                      postCrisis=4,
                      diffHor=5,
                      delWW=True)
    myData.getReady(name="New Models Data")

    # Pass 'myData' to the Experiment with the specified models.
    models = ["XGBoost", "LSTM"]
    exp = Experiment(myData, models, expType="InSample", rs=42)
    exp.run(n=1, disableTqdm=False)
    exp.rocGraph(save=True)
