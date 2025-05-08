from __future__ import annotations
import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

# --- Import model architecture and prep helper ---
from code.models.lstm_arch import build_lstm
from code.prep_helpers import reshape_for_lstm


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", message="Skipping features without any observed values", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="Do not pass an `input_shape`/`input_dim` argument to a layer", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ERI")

# ─────────── constants ───────────
TARGET = "crisisJST"
POST_CRISIS = "remove"
KEYS = ["iso", "year"]
SEED = 42
DEFAULT_MODEL_SAVE_DIR = Path("code/fitted_models")


# --- reshape_for_lstm function REMOVED from here ---


def preprocessor(num_cols: List[str]) -> ColumnTransformer:
    """median-impute + z-score scaling for numeric columns"""
    return ColumnTransformer(
        [("num",
          Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc",  StandardScaler())]),
          num_cols)],
        remainder="drop",
        verbose_feature_names_out=False
    )


def year_split(df: pd.DataFrame,
               holdout: float = .20) -> Tuple[pd.Index, pd.Index]:
    """Hold out the most recent `holdout` share of years."""
    if "year" not in df.columns:
        raise ValueError("Missing 'year' column.")
    df["year"] = pd.to_numeric(df["year"], errors='coerce')
    df = df.dropna(subset=["year"])
    if df.empty:
        raise ValueError("No valid 'year' data.")
    cut = np.quantile(df["year"], 1 - holdout)
    log.info(f"Splitting data at year {cut:.0f}")
    return df[df.year < cut].index, df[df.year >= cut].index


# ═══════════════════ main workflow ════════════════════
def main(panel: Path, out_eri: Path, plots: Path, model_save_dir: Path, show_for: str | None):
    # --- Load Data ---
    try:
        df = pd.read_csv(panel, low_memory=False)
        log.info("loaded %s (rows=%d, cols=%d)", panel, *df.shape)
    except Exception as e:
        log.error(f"Error loading panel: {e}", exc_info=True)
        sys.exit(1)

    # --- Clean Data ---
    if POST_CRISIS in df.columns:
        df = df[df[POST_CRISIS] == 0].copy()
        log.info("Post-crisis rows dropped.")
    else:
        log.info("No '%s' column, skipping post-crisis filter.", POST_CRISIS)
    if TARGET not in df.columns:
        log.warning("Target '%s' not found, assuming 0.", TARGET)
        df[TARGET] = 0
    else:
        df[TARGET] = df[TARGET].fillna(0).astype(int)

    # --- Features & Split ---
    num_cols = [c for c in df.columns if c not in KEYS +
                [TARGET, POST_CRISIS] and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        log.error("No numeric features found.")
        sys.exit(1)
    log.info(f"Found {len(num_cols)} numeric features.")
    X_raw, y = df[num_cols], df[TARGET]
    try:
        train_idx, test_idx = year_split(df)
    except ValueError as e:
        log.error(f"Year split error: {e}", exc_info=True)
        sys.exit(1)
    if train_idx.empty or test_idx.empty:
        log.error("Empty train/test split.")
        sys.exit(1)
    log.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    # --- Determine Feature Count ---
    temp_prep = preprocessor(num_cols)
    try:
        temp_prep.fit(X_raw.loc[train_idx])
        n_feat_after_prep = temp_prep.transform(
            X_raw.loc[[train_idx[0]]]).shape[1]
        log.info(f"Preprocessor outputs {n_feat_after_prep} features.")
    except Exception as e:
        log.error(f"Feature count determination failed: {e}", exc_info=True)
        sys.exit(1)
    del temp_prep

    # --- Model Definitions ---
    models: Dict[str, Pipeline] = {
        "Logit": Pipeline([("prep", preprocessor(num_cols)), ("est", LogisticRegression(max_iter=1500, random_state=SEED))]),
        "RandomForest": Pipeline([("prep", preprocessor(num_cols)), ("est", RandomForestClassifier(n_estimators=800, max_depth=8, random_state=SEED))]),
        "ExtraTrees": Pipeline([("prep", preprocessor(num_cols)), ("est", ExtraTreesClassifier(n_estimators=800, max_depth=8, random_state=SEED))]),
        "SVM": Pipeline([("prep", preprocessor(num_cols)), ("est", SVC(probability=True, gamma="scale", random_state=SEED))]),
        "MLP": Pipeline([("prep", preprocessor(num_cols)), ("est", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=SEED))]),
        "XGBoost": Pipeline([("prep", preprocessor(num_cols)), ("est", xgb.XGBClassifier(n_estimators=600, max_depth=5, learning_rate=.05, subsample=.8, colsample_bytree=.8, random_state=SEED, eval_metric="logloss"))]),

        "LSTM": Pipeline([
            ("prep", preprocessor(num_cols)),
            ("reshape", FunctionTransformer(reshape_for_lstm, validate=False)),
            # -----------------------------
            ("est", KerasClassifier(
                model=build_lstm,
                model__n_feat=n_feat_after_prep,
                epochs=40, batch_size=32, verbose=0,
                callbacks=[EarlyStopping(monitor="loss", patience=4, restore_best_weights=True)])
             )
        ])
    }

    # --- Fit & Evaluate ---
    tf.random.set_seed(SEED)
    aucs, probs_test, fitted_models = {}, {}, {}
    for name, pipe in models.items():
        log.info(f"Fitting {name} …")
        try:
            pipe.fit(X_raw.loc[train_idx], y.loc[train_idx])
            if name == "LSTM":
                log.debug("  Predicting with LSTM (manual transform)...")
                prep_step, reshape_step, keras_clf = pipe.named_steps[
                    'prep'], pipe.named_steps['reshape'], pipe.named_steps['est']
                X_test_prep = prep_step.transform(X_raw.loc[test_idx])
                X_test_reshaped = reshape_step.transform(X_test_prep)
                p_test = keras_clf.predict_proba(X_test_reshaped)[:, 1]
            else:
                p_test = pipe.predict_proba(X_raw.loc[test_idx])[:, 1]
            auc = roc_auc_score(y.loc[test_idx], p_test)
            aucs[name], probs_test[name], fitted_models[name] = auc, p_test, pipe
            log.info(f"  {name} AUC = {auc:.3f}")
        except Exception as e:
            log.error(f"Failed on {name}: {e}", exc_info=True)
            sys.exit(1)

    # --- Ensemble, Save, Plot ---
    if "XGBoost" in fitted_models and "LSTM" in fitted_models:
        log.info("Calculating ensemble and saving results...")
        top_two = sorted([(v, k) for k, v in aucs.items()
                         if k in ("XGBoost", "LSTM")], reverse=True)
        (auc1, m1), (auc2, m2) = top_two
        w1, w2 = auc1 / (auc1 + auc2) if (auc1 + auc2) > 0 else 0.5, 1.0 - \
            (auc1 / (auc1 + auc2) if (auc1 + auc2) > 0 else 0.5)
        log.info(f"Ensemble weights: {m1} {w1:.3f}, {m2} {w2:.3f}")
        eri_test = w1 * probs_test[m1] + w2 * probs_test[m2]

        log.info("Calculating training predictions for ensemble...")
        probs_train = {}
        for model_name in [m1, m2]:
            pipe = fitted_models[model_name]
            if model_name == "LSTM":
                prep_step, reshape_step, keras_clf = pipe.named_steps[
                    'prep'], pipe.named_steps['reshape'], pipe.named_steps['est']
                X_train_prep = prep_step.transform(X_raw.loc[train_idx])
                X_train_reshaped = reshape_step.transform(X_train_prep)
                probs_train[model_name] = keras_clf.predict_proba(X_train_reshaped)[
                    :, 1]
            else:
                probs_train[model_name] = pipe.predict_proba(
                    X_raw.loc[train_idx])[:, 1]
        eri_train = w1 * probs_train[m1] + w2 * probs_train[m2]

        df["eri"] = np.nan
        df.loc[train_idx, "eri"] = eri_train
        df.loc[test_idx,  "eri"] = eri_test

        # Save ERI
        try:
            out_eri.parent.mkdir(parents=True, exist_ok=True)
            df[KEYS + ["eri"]].dropna(subset=["eri"]
                                      ).to_csv(out_eri, index=False)
            log.info(
                f"ERI saved ({len(df.dropna(subset=['eri']))} rows) → {out_eri}")
        except Exception as e:
            log.error(f"Failed to save ERI: {e}", exc_info=True)

        # Save Models
        log.info(f"Saving fitted models to {model_save_dir}...")
        try:
            model_save_dir.mkdir(exist_ok=True, parents=True)
            for name in ["XGBoost", "LSTM"]:  # Only save ensemble components
                filename = model_save_dir / f"{name}_pipeline.joblib"
                joblib.dump(fitted_models[name], filename)
                log.info(f"  Saved {name} pipeline to {filename}")
        except Exception as e:
            # Log full error
            log.error(f"Failed saving models: {e}", exc_info=True)
        try:
            plots.mkdir(parents=True, exist_ok=True)
            fpr1, tpr1, _ = roc_curve(y.loc[test_idx], probs_test[m1])
            fpr2, tpr2, _ = roc_curve(y.loc[test_idx], probs_test[m2])
            fprE, tprE, _ = roc_curve(y.loc[test_idx], eri_test)
            ensemble_auc = roc_auc_score(y.loc[test_idx], eri_test)
            plt.figure(figsize=(6, 6))
            plt.plot(fpr1, tpr1, label=f"{m1} (AUC={auc1:.3f})", alpha=0.8)
            plt.plot(fpr2, tpr2, label=f"{m2} (AUC={auc2:.3f})", alpha=0.8)
            plt.plot(fprE, tprE, "--", lw=2.5, color='red',
                     label=f"Ensemble ERI (AUC={ensemble_auc:.3f})")
            plt.plot([0, 1], [0, 1], "k:", alpha=.6, label="Random (AUC=0.5)")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves (Test Set)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots / "roc_ensemble.png", dpi=300)
            plt.close()
            log.info(f"  Saved ROC curve plot.")
            plt.figure(figsize=(7, 5))
            sns.histplot(df["eri"].dropna(), kde=True, bins=30)
            plt.title("ERI Score Distribution")
            plt.xlabel("ERI Score")
            plt.tight_layout()
            plt.savefig(plots / "eri_hist.png", dpi=300)
            plt.close()
            log.info(f"  Saved ERI histogram.")
            log.info(f"Plots saved → {plots}")
        except Exception as e:
            log.error(f"Plotting failed: {e}", exc_info=True)

    else:
        log.warning("Ensemble components missing, skipping final steps.")

    # --- Optional Timeline ---
    if show_for:
        iso = show_for.upper()
        if "eri" in df.columns:
            ts = df[df.iso == iso].sort_values("year")[["year", "eri", TARGET]]
        else:
            ts = pd.DataFrame()  # Avoid error if eri column missing
        if ts.empty:
            log.warning(f"No rows/ERI for iso={iso}")
        else:
            log.info(f"--- ERI timeline for {iso} ---\nYear   ERI   Crisis\n" + "\n".join(
                f"{int(r.year)}  {r.eri:.3f}    {int(r[TARGET])}" for _, r in ts.iterrows()))


# ═══════════════════ CLI ════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train ERI ensemble.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--panel",    required=True, type=Path,
                   help="Input panel data CSV.")
    p.add_argument("--out-eri",  required=True, type=Path,
                   help="Output path for ERI scores CSV.")
    p.add_argument("--plots",    default=Path("plots"),
                   type=Path, help="Directory for output plots.")
    p.add_argument("--eri-for",  type=str,
                   help="ISO-3 code to print ERI timeline for.")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_SAVE_DIR,
                   type=Path, help="Directory to save fitted models.")
    args = p.parse_args()
    main(args.panel, args.out_eri, args.plots, args.model_dir.resolve(),
         args.eri_for)
