Supplementary code and data
===========================
44_ML_Updated.xlsx               44 operating conditions (HC values corrected). Inputs: fuel, injection timing, load.
                                 Responses: BTE, BSFC [kg/kWh; converted to g/kWh in the script], CO, HC, NOx.
engine_ml_reanalysis.py          Complete analysis: descriptive statistics, one-hot encoding, hold-out split, repeated nested CV
                                 (10 x 4 fuel-stratified folds, inner 5-fold GridSearchCV), baselines, leave-boundary-timing-out test,
                                 bootstrap CIs, Bland-Altman, corrected resampled t-test, factorial ANOVA, SHAP, Figures 4-10.
make_figure2_workflow.py         Figure 2
make_figure3_distributions.py    Figure 3
requirements.txt                 Package versions of the reference run (Python 3.12; identical results reproduced with Python 3.13)
results/results.json             All numerical results (every value in Tables 3-11 and Sections 4.1-4.6)

Run:  pip install -r requirements.txt
      python engine_ml_reanalysis.py 44_ML_Updated.xlsx results
Runtime about 25 min on one CPU core. Fold results are cached in results/folds/ so an interrupted run resumes
(delete results/ before re-running with a changed dataset). Random seed 42 throughout.
