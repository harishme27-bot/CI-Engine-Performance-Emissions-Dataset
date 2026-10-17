"""
engine_ml_reanalysis.py
Reproducible re-analysis for:
"Comparative Analysis of XGBoost and Support Vector Machine Models for Predicting
Performance and Emissions in Single-Cylinder Compression Ignition Engines"

Addresses third-round reviewer comments:
  - one-hot fuel encoding inside the CV pipeline (C4)
  - repeated nested stratified CV (C2, C7), holdout reported as secondary with real n_test
  - no early stopping; n_estimators tuned in inner CV (C14)
  - baselines: mean predictor, MLR, quadratic ridge, random forest (C17)
  - leave-boundary-timing-out extrapolation test (C3)
  - Bland-Altman agreement analysis (C5)
  - factorial ANOVA on the balanced biodiesel sub-design + fuel x load at 23 deg (C6)
  - SHAP global importance with repeated-CV stability (C12, C13)
  - sMAPE alongside MAPE (C15), BSFC in g/kWh everywhere (C8)
  - descriptive statistics / exact design counts from one authoritative run (C9-C11)

Usage:  python engine_ml_reanalysis.py 44_ML_Updated.xlsx  [outdir]
"""
import sys, os, json, time, warnings, platform
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import sklearn, xgboost, shap, statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold,
                                     RepeatedStratifiedKFold)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
SEED = 42
DATA = sys.argv[1] if len(sys.argv) > 1 else "44_ML_Updated.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(OUT, exist_ok=True)
N_OUTER, N_INNER, N_BOOT = 4, 5, 2000   # 4 outer folds -> exactly one diesel condition per test fold
N_REPEATS = int(os.environ.get("N_REPEATS", 10))

# ------------------------------------------------------------------ data
df = pd.read_excel(DATA)
df.columns = df.columns.str.strip()
df = df.rename(columns={"Injection Timing (deg bTDC)": "Timing", "Load (%)": "Load",
                        "BTE (%)": "BTE", "BSFC (kg/kWh)": "BSFC", "CO (%)": "CO",
                        "HC (ppm)": "HC", "NOx (ppm)": "NOx"})
df["BSFC"] = df["BSFC"] * 1000.0          # kg/kWh -> g/kWh (single unit throughout)
FUELS = ["Diesel", "MESO-B20", "MEWVO-B20"]
TARGETS = ["BTE", "BSFC", "CO", "HC", "NOx"]
UNITS = {"BTE": "%", "BSFC": "g/kWh", "CO": "% vol", "HC": "ppm", "NOx": "ppm"}
FEATS = ["Fuel", "Timing", "Load"]
X = df[FEATS].copy()
strata = df["Fuel"].values
R = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "shap": shap.__version__,
                     "statsmodels": statsmodels.__version__, "seed": SEED}}

# ------------------------------------------------------------------ descriptive
desc = df[["Timing", "Load"] + TARGETS].describe().T
desc["skew"] = df[["Timing", "Load"] + TARGETS].skew()          # adjusted Fisher-Pearson
desc.to_csv(f"{OUT}/table3_descriptive.csv")
counts = pd.crosstab([df.Fuel, df.Timing], df.Load)
counts.to_csv(f"{OUT}/design_counts.csv")
R["fuel_counts"] = df.Fuel.value_counts().to_dict()
R["timing_counts"] = {int(k): int(v) for k, v in df.Timing.value_counts().sort_index().items()}
R["load_counts"] = {int(k): int(v) for k, v in df.Load.value_counts().sort_index().items()}
corr = df[["Timing", "Load"] + TARGETS].corr(method="pearson")
corr.to_csv(f"{OUT}/correlation_pearson.csv")
# IQR outlier flags (statistical, not physical)
out_flags = {}
for c in TARGETS:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    m = (df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)
    out_flags[c] = df.loc[m, ["Fuel", "Timing", "Load", c]].to_dict("records")
R["iqr_outliers"] = out_flags

# ------------------------------------------------------------------ models
def prep(scale, drop=None):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("num", num, ["Timing", "Load"]),
                              ("fuel", OneHotEncoder(categories=[FUELS], drop=drop,
                                                     handle_unknown="ignore"), ["Fuel"])])

inner_cv = KFold(N_INNER, shuffle=True, random_state=SEED)
SCORING = "neg_root_mean_squared_error"

def make_xgb():
    pipe = Pipeline([("prep", prep(False)),
                     ("model", XGBRegressor(objective="reg:squarederror", subsample=0.8,
                                            colsample_bytree=1.0, min_child_weight=1,
                                            gamma=0.0, tree_method="exact",
                                            random_state=SEED, n_jobs=1))])
    grid = {"model__n_estimators": [200, 500], "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.05, 0.1], "model__reg_lambda": [1.0, 5.0]}
    return GridSearchCV(pipe, grid, cv=inner_cv, scoring=SCORING, n_jobs=1)

def make_svr():
    pipe = TransformedTargetRegressor(
        regressor=Pipeline([("prep", prep(True)), ("model", SVR())]),
        transformer=StandardScaler())
    p = "regressor__model__"
    grid = [{p + "kernel": ["rbf"], p + "C": [1, 10, 100, 1000],
             p + "gamma": [0.01, 0.1, 0.5, 1.0], p + "epsilon": [0.01, 0.05, 0.1]},
            {p + "kernel": ["poly"], p + "degree": [2], p + "coef0": [1.0], p + "gamma": ["scale"],
             p + "C": [1, 10, 100], p + "epsilon": [0.01, 0.05, 0.1]},
            {p + "kernel": ["linear"], p + "C": [0.1, 1, 10], p + "epsilon": [0.01, 0.05, 0.1]}]
    return GridSearchCV(pipe, grid, cv=inner_cv, scoring=SCORING, n_jobs=1)

def make_mean(): return DummyRegressor(strategy="mean")
def make_mlr(): return Pipeline([("prep", prep(True, "first")), ("model", LinearRegression())])
def make_quad():
    return Pipeline([("prep", prep(True, "first")),
                     ("poly", PolynomialFeatures(2, include_bias=False)),
                     ("model", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
def make_rf():
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=1, random_state=SEED, n_jobs=1)

class RFWrap:  # RF needs numeric input -> use prep
    pass
def make_rf_pipe(): return Pipeline([("prep", prep(False)), ("model", make_rf())])

MODELS = {"XGBoost": make_xgb, "SVR": make_svr, "Mean": make_mean, "MLR": make_mlr,
          "QuadRidge": make_quad, "RF": make_rf_pipe}

def smape(y, p): return 100 * np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p)))
def mape(y, p): return 100 * np.mean(np.abs((y - p) / y))
def metrics(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=float(np.sqrt(mean_squared_error(y, p))), MAPE=mape(y, p), sMAPE=smape(y, p))

def best_params(est):
    return {k.split("__")[-1]: v for k, v in est.best_params_.items()} if hasattr(est, "best_params_") else {}

# ------------------------------------------------------------------ SHAP helpers
def shap_xgb(fitted_gs, Xe):
    pipe = fitted_gs.best_estimator_
    Xt = pipe.named_steps["prep"].transform(Xe)
    sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xt)
    # columns: Timing, Load, Fuel_Diesel, Fuel_MESO, Fuel_MEWVO -> aggregate fuel (additivity)
    return np.column_stack([sv[:, 0], sv[:, 1], sv[:, 2:].sum(1)])

def shap_svr(fitted_gs, Xb, Xe):
    est = fitted_gs.best_estimator_
    code = {f: i for i, f in enumerate(FUELS)}
    def f(A):
        D = pd.DataFrame({"Timing": A[:, 0], "Load": A[:, 1],
                          "Fuel": [FUELS[int(round(v))] for v in A[:, 2]]})
        return est.predict(D)
    enc = lambda D: np.column_stack([D.Timing, D.Load, D.Fuel.map(code)]).astype(float)
    ex = shap.KernelExplainer(f, enc(Xb))
    return ex.shap_values(enc(Xe), nsamples="auto", silent=True)  # 3 features -> exact enumeration

SHAP_ORDER = ["Timing", "Load", "Fuel"]

# ------------------------------------------------------------------ 1. holdout (secondary)
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, stratify=strata, random_state=SEED)
R["holdout"] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                "test_fuels": df.Fuel.iloc[te].value_counts().to_dict(),
                "test_rows": df.iloc[te][FEATS].to_dict("records")}
rng = np.random.default_rng(SEED)
hold = {}
_hc = f"{OUT}/holdout_cache.json"
if os.path.exists(_hc):
    hold = {tuple(k.split("|")): v for k, v in json.load(open(_hc)).items()}
for t in ([] if hold else TARGETS):
    y = df[t].values
    for name in ["XGBoost", "SVR"]:
        m = MODELS[name]()
        t0 = time.perf_counter(); m.fit(X.iloc[tr], y[tr]); tune_t = time.perf_counter() - t0
        p = m.predict(X.iloc[te])
        met = metrics(y[te], p)
        bs = {"R2": [], "RMSE": []}
        for _ in range(N_BOOT):
            b = rng.integers(0, len(te), len(te))
            if np.ptp(y[te][b]) == 0: continue
            bs["R2"].append(r2_score(y[te][b], p[b])); bs["RMSE"].append(np.sqrt(mean_squared_error(y[te][b], p[b])))
        ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
        # timing of final configuration
        final = clone(m.best_estimator_)
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); final.fit(X.iloc[tr], y[tr]); ts.append(time.perf_counter() - t0)
        tp = []
        for _ in range(200):
            t0 = time.perf_counter(); final.predict(X.iloc[te]); tp.append(time.perf_counter() - t0)
        hold[(t, name)] = dict(**met, CI=ci, best=best_params(m), grid_time_s=tune_t,
                               fit_time_ms=1000 * np.median(ts),
                               predict_time_ms_per_sample=1000 * np.median(tp) / len(te),
                               y_test=y[te].tolist(), y_pred=p.tolist())
R["holdout_results"] = {f"{t}|{n}": v for (t, n), v in hold.items()}
json.dump(R["holdout_results"], open(f"{OUT}/holdout_cache.json", "w"), default=float)
print("holdout done", flush=True)

# ------------------------------------------------------------------ 2. repeated nested CV (primary)
outer = RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=SEED)
splits = list(outer.split(X, strata))
oof = {(t, m): np.full((N_REPEATS, len(df)), np.nan) for t in TARGETS for m in MODELS}
fold_r2 = {(t, m): [] for t in TARGETS for m in MODELS}
fold_rmse = {(t, m): [] for t in TARGETS for m in MODELS}
shap_imp = {(t, m): [] for t in TARGETS for m in ["XGBoost", "SVR"]}
chosen = {(t, m): [] for t in TARGETS for m in ["XGBoost", "SVR"]}
t_start = time.time()
import pickle
os.makedirs(f"{OUT}/folds", exist_ok=True)
for k, (a, b) in enumerate(splits):
    rep = k // N_OUTER
    ck = f"{OUT}/folds/fold_{k:02d}.pkl"
    if os.path.exists(ck):
        rec = pickle.load(open(ck, "rb"))
    else:
        rec = {}
        for t in TARGETS:
            y = df[t].values
            for name, mk in MODELS.items():
                m = mk(); m.fit(X.iloc[a], y[a]); p = m.predict(X.iloc[b])
                entry = {"p": p}
                if name == "XGBoost":
                    sv = shap_xgb(m, X.iloc[b]); entry["params"] = best_params(m)
                elif name == "SVR":
                    sv = shap_svr(m, X.iloc[a], X.iloc[b]); entry["params"] = best_params(m)
                else:
                    sv = None
                if sv is not None:
                    imp = np.abs(sv).mean(0); entry["shap"] = 100 * imp / imp.sum()
                rec[(t, name)] = entry
        pickle.dump(rec, open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
    for (t, name), e in rec.items():
        y = df[t].values
        oof[(t, name)][rep, b] = e["p"]
        fold_r2[(t, name)].append(r2_score(y[b], e["p"]))
        fold_rmse[(t, name)].append(np.sqrt(mean_squared_error(y[b], e["p"])))
        if "shap" in e:
            shap_imp[(t, name)].append(e["shap"]); chosen[(t, name)].append(e["params"])
    print(f"outer fold {k+1}/{len(splits)}  {time.time()-t_start:.0f}s", flush=True)

cv = {}
for (t, name), P in oof.items():
    y = df[t].values
    per_rep = pd.DataFrame([metrics(y, P[r]) for r in range(N_REPEATS)])
    cv[(t, name)] = {k: dict(mean=float(per_rep[k].mean()), sd=float(per_rep[k].std(ddof=1)),
                             p2_5=float(per_rep[k].quantile(.025)), p97_5=float(per_rep[k].quantile(.975)))
                     for k in per_rep}
    cv[(t, name)]["fold_R2_mean"] = float(np.mean(fold_r2[(t, name)]))
    cv[(t, name)]["fold_R2_sd"] = float(np.std(fold_r2[(t, name)], ddof=1))
R["cv_results"] = {f"{t}|{n}": v for (t, n), v in cv.items()}

# Nadeau-Bengio corrected resampled t-test on fold RMSE, XGBoost vs SVR
def corrected_t(d, n_train, n_test):
    d = np.asarray(d); J = len(d); mean = d.mean(); var = d.var(ddof=1)
    tstat = mean / np.sqrt((1 / J + n_test / n_train) * var)
    return float(tstat), float(2 * stats.t.sf(abs(tstat), J - 1))
nte = len(df) / N_OUTER; ntr = len(df) - nte
R["xgb_vs_svr"] = {}
for t in TARGETS:
    d = np.array(fold_rmse[(t, "XGBoost")]) - np.array(fold_rmse[(t, "SVR")])
    tt, pp = corrected_t(d, ntr, nte)
    R["xgb_vs_svr"][t] = dict(mean_diff_RMSE=float(d.mean()), t=tt, p=pp,
                              frac_folds_xgb_better=float(np.mean(d < 0)))

# SHAP stability
R["shap_cv"] = {f"{t}|{m}": dict(zip(SHAP_ORDER, [dict(mean=float(np.mean(v, 0)[i]), sd=float(np.std(v, 0, ddof=1)[i]))
                                                    for i in range(3)])) for (t, m), v in shap_imp.items()}
# most frequently chosen hyper-parameters
def mode_params(lst):
    s = pd.Series([json.dumps(d, sort_keys=True, default=str) for d in lst]).value_counts()
    return dict(most_common=json.loads(s.index[0]), freq=int(s.iloc[0]), of=len(lst))
R["chosen_params"] = {f"{t}|{m}": mode_params(v) for (t, m), v in chosen.items()}

# ------------------------------------------------------------------ 3. Bland-Altman on repeat-averaged OOF
ba = {}
for t in TARGETS:
    y = df[t].values
    for name in ["XGBoost", "SVR"]:
        p = np.nanmean(oof[(t, name)], 0); d = y - p; n = len(d)
        bias, sd = d.mean(), d.std(ddof=1); se = sd / np.sqrt(n); tc = stats.t.ppf(.975, n - 1)
        ttest = stats.ttest_rel(y, p); wil = stats.wilcoxon(y, p)
        # proportional bias: slope of difference on mean
        slope = stats.linregress((y + p) / 2, d)
        ba[(t, name)] = dict(bias=float(bias), bias_CI=[float(bias - tc * se), float(bias + tc * se)],
                             LoA=[float(bias - 1.96 * sd), float(bias + 1.96 * sd)],
                             within_LoA_pct=float(100 * np.mean(np.abs(d - bias) <= 1.96 * sd)),
                             prop_bias_slope=float(slope.slope), prop_bias_p=float(slope.pvalue),
                             t_p=float(ttest.pvalue), wilcoxon_p=float(wil.pvalue))
R["bland_altman"] = {f"{t}|{n}": v for (t, n), v in ba.items()}

# ------------------------------------------------------------------ 4. leave-boundary-timing-out
inner_mask = df.Timing.isin([19, 21, 23]).values
lbo = {}
for t in TARGETS:
    y = df[t].values
    for name in ["XGBoost", "SVR", "MLR", "QuadRidge", "RF", "Mean"]:
        m = MODELS[name](); m.fit(X[inner_mask], y[inner_mask]); p = m.predict(X[~inner_mask])
        lbo[(t, name)] = metrics(y[~inner_mask], p)
R["boundary_holdout"] = {"n_train": int(inner_mask.sum()), "n_test": int((~inner_mask).sum()),
                         "results": {f"{t}|{n}": v for (t, n), v in lbo.items()}}

# ------------------------------------------------------------------ 5. factorial ANOVA
bio = df[df.Fuel != "Diesel"].copy()
d23 = df[df.Timing == 23].copy()
anova = {}
for t in TARGETS:
    f1 = smf.ols(f"{t} ~ C(Fuel) + C(Timing) + C(Load) + C(Fuel):C(Timing) + C(Fuel):C(Load) + C(Timing):C(Load)", bio).fit()
    a1 = anova_lm(f1, typ=2); a1["eta_p2"] = a1.sum_sq / (a1.sum_sq + a1.loc["Residual", "sum_sq"])
    f2 = smf.ols(f"{t} ~ C(Fuel) + C(Load)", d23).fit()
    a2 = anova_lm(f2, typ=2); a2["eta_p2"] = a2.sum_sq / (a2.sum_sq + a2.loc["Residual", "sum_sq"])
    anova[t] = {"biodiesel_factorial": a1.round(6).reset_index().to_dict("records"),
                "timing23_fuel_load": a2.round(6).reset_index().to_dict("records")}
    # fuel means at 23 deg
    anova[t]["means_23deg"] = d23.groupby("Fuel")[t].mean().to_dict()
R["anova"] = anova

# ------------------------------------------------------------------ 6. final models on all data: SHAP dependence
final_shap = {}
for t in ["NOx", "BTE"]:
    m = make_xgb(); m.fit(X, df[t].values); final_shap[t] = shap_xgb(m, X)

# ------------------------------------------------------------------ save
def conv(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=conv)
np.savez(f"{OUT}/oof.npz", **{f"{t}|{m}": v for (t, m), v in oof.items()})

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
COL = {"XGBoost": "#1f5aa6", "SVR": "#c0392b"}
LAB = {"BTE": "BTE (%)", "BSFC": "BSFC (g/kWh)", "CO": "CO (% vol)", "HC": "HC (ppm)", "NOx": "NOx (ppm)"}

# Fig A: repeated nested CV R2 by model
order = ["Mean", "MLR", "QuadRidge", "RF", "XGBoost", "SVR"]
fig, axs = plt.subplots(1, 5, figsize=(13, 3.2), sharey=False)
for ax, t in zip(axs, TARGETS):
    y = df[t].values
    data = [[r2_score(y, oof[(t, m)][r]) for r in range(N_REPEATS)] for m in order]
    bp = ax.boxplot(data, widths=.6, patch_artist=True)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(COL.get(m, "#bbbbbb")); patch.set_alpha(.75)
    ax.set_xticks(range(1, 7)); ax.set_xticklabels(["Mean", "MLR", "Quad", "RF", "XGB", "SVR"], rotation=45)
    ax.set_title(LAB[t]); ax.axhline(0, color="k", lw=.5); ax.grid(alpha=.3)
    lo = min(-0.1, min(min(d) for d in data[1:]) - .05); ax.set_ylim(lo, 1.02)
axs[0].set_ylabel("Out-of-fold R² (10 repeats × 4-fold)")
plt.tight_layout(); plt.savefig(f"{OUT}/Fig_cv_r2_baselines.png", dpi=300); plt.close()

# Fig B/C: parity and residuals (repeat-averaged OOF)
for kind in ["parity", "residual"]:
    fig, axs = plt.subplots(2, 5, figsize=(14, 5.6))
    for i, name in enumerate(["XGBoost", "SVR"]):
        for j, t in enumerate(TARGETS):
            ax = axs[i, j]; y = df[t].values; P = oof[(t, name)]; p = P.mean(0); s = P.std(0)
            mk = {"Diesel": "s", "MESO-B20": "o", "MEWVO-B20": "^"}
            for f in FUELS:
                mm = (df.Fuel == f).values
                if kind == "parity":
                    ax.errorbar(y[mm], p[mm], yerr=s[mm], fmt=mk[f], ms=4, color=COL[name], alpha=.8,
                                elinewidth=.6, capsize=1.5, label=f)
                else:
                    ax.scatter(p[mm], y[mm] - p[mm], marker=mk[f], s=16, color=COL[name], alpha=.8, label=f)
            if kind == "parity":
                lim = [min(y.min(), p.min()), max(y.max(), p.max())]; ax.plot(lim, lim, "k--", lw=.8)
                ax.set_xlabel(f"Measured {LAB[t]}"); ax.set_ylabel(f"Predicted {LAB[t]}")
                r2 = cv[(t, name)]["R2"]["mean"]; ax.text(.04, .9, f"R² = {r2:.3f}", transform=ax.transAxes)
            else:
                ax.axhline(0, color="k", lw=.8); ax.set_xlabel(f"Predicted {LAB[t]}"); ax.set_ylabel("Residual")
            ax.set_title(f"{name}: {t}"); ax.grid(alpha=.3)
    axs[0, 0].legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{OUT}/Fig_{kind}_oof.png", dpi=300); plt.close()

# Fig D: Bland-Altman
fig, axs = plt.subplots(2, 5, figsize=(14, 5.6))
for i, name in enumerate(["XGBoost", "SVR"]):
    for j, t in enumerate(TARGETS):
        ax = axs[i, j]; y = df[t].values; p = np.nanmean(oof[(t, name)], 0); b = ba[(t, name)]
        ax.scatter((y + p) / 2, y - p, s=14, color=COL[name], alpha=.8)
        ax.axhline(b["bias"], color="k", lw=1)
        for L in b["LoA"]: ax.axhline(L, color="k", ls="--", lw=.8)
        ax.set_title(f"{name}: {t}"); ax.set_xlabel(f"Mean of measured & predicted {LAB[t]}")
        ax.set_ylabel("Measured − predicted"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/Fig_bland_altman.png", dpi=300); plt.close()

# Fig E: SHAP importance with CV stability
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
w = .25
for ax, name in zip(axs, ["XGBoost", "SVR"]):
    for k, f in enumerate(SHAP_ORDER):
        mu = [R["shap_cv"][f"{t}|{name}"][f]["mean"] for t in TARGETS]
        sd = [R["shap_cv"][f"{t}|{name}"][f]["sd"] for t in TARGETS]
        ax.bar(np.arange(5) + (k - 1) * w, mu, w, yerr=sd, capsize=2, label=f,
               color=["#4c72b0", "#dd8452", "#55a868"][k])
    ax.set_xticks(range(5)); ax.set_xticklabels(TARGETS); ax.set_title(f"{name}: share of mean |SHAP| (%)")
    ax.grid(axis="y", alpha=.3)
axs[0].set_ylabel("Global importance (%), mean ± SD over 40 CV folds"); axs[0].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/Fig_shap_importance_cv.png", dpi=300); plt.close()

# Fig F: SHAP dependence (final XGBoost fitted on all 44)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, (t, fi, fname) in zip(axs, [("NOx", 0, "Timing"), ("BTE", 1, "Load")]):
    sv = final_shap[t]; xv = df[fname].values
    sc = ax.scatter(xv + np.random.default_rng(1).uniform(-.3, .3, len(xv)) * (1 if fname == "Timing" else 3),
                    sv[:, fi], c=df["Load" if fname == "Timing" else "Timing"], cmap="viridis", s=20)
    plt.colorbar(sc, ax=ax, label="Load (%)" if fname == "Timing" else "Timing (° bTDC)")
    ax.set_xlabel("Injection timing (° bTDC)" if fname == "Timing" else "Engine load (%)")
    ax.set_ylabel(f"SHAP value for {LAB[t]}"); ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.3)
    ax.set_title(f"XGBoost SHAP dependence: {fname} → {t}")
plt.tight_layout(); plt.savefig(f"{OUT}/Fig_shap_dependence.png", dpi=300); plt.close()

# Fig G: correlation heatmap (numeric variables only)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
lbl = ["Timing", "Load", "BTE", "BSFC", "CO", "HC", "NOx"]
ax.set_xticklabels(lbl, rotation=45); ax.set_yticklabels(lbl)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, label="Pearson r"); plt.tight_layout()
plt.savefig(f"{OUT}/Fig_correlation.png", dpi=300); plt.close()
print("DONE", flush=True)
