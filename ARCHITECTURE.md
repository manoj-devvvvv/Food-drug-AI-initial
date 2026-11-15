# Food–Drug Interaction ML with SHAP

**Team:** Team BuildQ
**Project:** Food–Drug ML model with SHAP
**Purpose:** predict whether a drug and a food compound should be avoided together,
taken together, or are non-significant — with calibrated probabilities, dataset
evidence, and SHAP explanations.

This document explains how the system was built, how it is served, and which
technologies it uses. It complements the per-brick reports in `reports/` and the
notebooks in `notebooks/`.

---

## 1. What the system does

Input: a drug and a food compound, supplied as a name, database ID or SMILES.
Output:

- one of three classes — `0` avoid together, `1` beneficial / take with food,
  `2` non-significant;
- class probabilities and a confidence value when the ML model answers;
- a plain-language sentence and, where applicable, the clinical rule or the exact
  dataset annotation behind the answer;
- abstention (`uncertain` / `not_supported`) when the model is unsure, the drug is
  outside its applicability domain, or the compound is inorganic.

Every response carries the disclaimer *“Research tool; confirm with a physician or
pharmacist.”*

---

## 2. Repository at a glance

```
dfi/
├── config.yaml                  paths, cutoffs, model and label settings
├── requirements.txt             pinned Python environment
├── ARCHITECTURE.md              this document
├── src/                         pipeline (bricks) and inference engine
│   ├── common.py                config and JSON/Parquet helpers
│   ├── brick1_audit.py          raw data audit
│   ├── brick2_clean_drugs.py    drug cleaning and deduplication
│   ├── brick3_clean_foods.py    compound cleaning, Tanimoto dedup
│   ├── brick4_labels.py         label construction (Modes P and S)
│   ├── features.py              60 MOE descriptors and 3,780 pair features
│   ├── brick5_features.py       descriptor computation
│   ├── brick6_splits.py         Split G / Split R
│   ├── brick7_selection.py      four-step selection to 18 features
│   ├── brick8_models.py         five classifiers, tuning, frozen winner
│   ├── brick8b_compare.py       all five models on held-out test data
│   ├── brick9_evaluate.py       test evaluation, threshold sweep, per-drug spread
│   ├── brick10_shap.py          TreeSHAP per class and mode
│   ├── brick11_recommender.py   clinical case suite
│   ├── brick12_enhancements.py  ablations, calibration, conformal, bootstrap CIs
│   └── recommend.py             production inference engine
├── models/                      selected features, frozen models, train-drug lists
├── data/interim|processed       cleaned tables, features, splits
├── reports/                     JSON, CSV, Markdown artifacts per brick
├── figures/                     confusion matrices, SHAP plots, comparison charts
├── notebooks/                   00–13 walkthrough notebooks + 08c comparison
├── scripts/                     run_all.sh, check_gates.py, make_report.py
└── web/
    ├── app.py                   FastAPI service
    └── static/
        ├── landing.html         landing page
        ├── index.html           predictor page
        ├── about.html           project documentation page
        ├── styles.css           shared design system and predictor styles
        ├── pages.css            landing/about layout
        └── app.js               predictor behaviour
```

---

## 3. End-to-end development flow

| Stage | Script | Main artifacts | Key result |
|---|---|---|---|
| Environment | — | `reports/brick0_environment.md` | Python, RDKit, XGBoost, LightGBM, SHAP verified |
| Data audit | `brick1_audit.py` | `brick1_audit.json/md` | 1,516 drugs, 23,950 interactions, 838,492 food links |
| Clean drugs | `brick2_clean_drugs.py` | `brick2_funnel.*`, `drugs_clean.parquet` | 1,359 unique drugs with valid structures |
| Clean foods | `brick3_clean_foods.py` | `brick3_funnel.*`, `foods_clean.parquet` | 4,877 organic compounds after Tanimoto dedup |
| Labels | `brick4_labels.py` | `pairs_labeled.parquet`, `brick4_labels.md` | Mode S 21,042 pairs; Mode P 407,699 pairs |
| Features | `brick5_features.py` | `drug_moe.parquet`, `food_moe.parquet` | 60 MOE descriptors per molecule, all finite |
| Splits | `brick6_splits.py` | `splits_S.parquet`, `splits_P.parquet` | Drug-grouped Split G and random Split R |
| Selection | `brick7_selection.py` | `selected_features_{S,P}.json` | 3,780 → 18 features |
| Models | `brick8_models.py` | `best_model_{S,P}.joblib`, tuning CSVs | LightGBM wins both modes |
| Comparison | `brick8b_compare.py` | `model_comparison.*` | All five models on held-out test |
| Evaluation | `brick9_evaluate.py` | `metrics_{S,P}.json`, predictions | Split G headline metrics |
| Explainability | `brick10_shap.py` | `shap_summary_*` figures | Per-class TreeSHAP attributions |
| Case suite | `brick11_recommender.py` | `brick11_case_study.*` | 20 cases, 17/17 scored pass |
| Enhancements | `brick12_enhancements.py` | `brick12_enhancements.*` | Ablations, conformal, bootstrap CIs |
| Report | `make_report.py` | `reports/report.md` | Consolidated narrative |
| Web app | `web/app.py` | running FastAPI service | Landing, predictor, about, API |

The full pipeline can be regenerated with `scripts/run_all.sh`; `scripts/check_gates.py`
verifies 31 gates across artifacts and reports.

---

## 4. Data and labels

### 4.1 Raw and cleaned data

| Stage | Count |
|---|---:|
| Raw drugs | 1,516 |
| Raw interaction records | 23,950 |
| Raw food–compound links | 838,492 |
| Unique raw compounds | 13,282 |
| Clean drugs | 1,359 |
| Clean organic compounds | 4,877 |
| Inorganic compounds (excluded) | 145 |
| Retained food–compound links | 152,320 |

Drugs are cleaned to valid, deduplicated structures (InChIKey). Compounds are
deduplicated with canonical SMILES and a Morgan/Tanimoto near-duplicate filter,
while protected marker compounds (ethanol, tyramine, caffeine, bergamottin, naringin,
hyperforin, phylloquinone, vitamin K1, dimethyl disulfide) are always retained.

### 4.2 Two label modes

The source data is record-level free text, so turning it into pair-level training
labels requires a policy.

| | Specific (Mode S) | Paper-like (Mode P) |
|---|---|---|
| Label granularity | one per drug–compound pair | one per drug, copied to every sampled compound |
| Compound selection | named in the text, curated food markers, or food-specific compounds | all sampled organic compounds |
| Semantics | corrected: warning-style “Positive” → avoid | paper semantics, kept for comparability |
| Drugs covered | 444 | 1,359 |
| Pairs | 21,042 | 407,699 |
| Class counts | `{0: 4,898, 1: 24, 2: 16,120}` | `{0: 229,499, 1: 65,100, 2: 113,100}` |
| Synthetic weak pairs | 11,517 (weight 0.5) | none |
| Intended use | trustworthy prediction | reproduction / comparability |

Mode S is the default and the mode used for the headline results. Mode P is
explicitly leakage-prone because a single drug-level warning is imposed on every
food compound, including compounds that were never studied.

---

## 5. Feature engineering

1. **Molecule → descriptors.** RDKit parses canonical SMILES; PyBioMed computes 60
   MOE descriptors per molecule (VSA, EState, Labute ASA, slogP and related terms).
2. **Pair → feature vector.** For each pair the 60 descriptor values of the drug and
   the compound are combined into:
   - 3,600 products `drug_i * food_j`
   - 60 sums `drug_i + food_i`
   - 60 absolute differences `|drug_i - food_i|`
   - 60 ratios `drug_i / food_i`
   - total: **3,780 candidate features**
3. **Selection.** Four steps on a stratified 10,495-pair training sample:
   - variance filter (threshold 0.8): 3,780 → 3,521
   - Pearson correlation pruning (|r| ≥ 0.75): → 1,126
   - LASSO (α = 0.01) non-zero coefficients: → 174
   - RFECV / RFE to a fixed 18 features: → **18**
   Stability selection runs the selection three times; the final set is persisted in
   `models/selected_features_{mode}.json`.

Example Mode S features include `PEOEVSA5+PEOEVSA5`, `VSAEstate6+VSAEstate6`,
`slogPVSA1/slogPVSA1`, `MRVSA9/MRVSA9`, `abs(MRVSA5-MRVSA5)`. The exact list is
loaded at inference time, so the serving code cannot drift from training.

---

## 6. Splits and validation protocol

- **Split G (headline).** Whole drugs are assigned to train/validation/test, so test
  drugs never appear during training. This measures generalization to unseen drugs.
  Mode S: 10,537 train / 7,872 validation / 2,633 test pairs (2,618 after feature
  filtering). Mode P: 204,000 / 152,999 / 50,700.
- **Split R (reference).** Random pair split with the paper's class proportions; used
  for comparability and to expose how much easier random-split evaluation is.
- **Grouped CV.** During tuning, `GroupKFold` groups by drug on Split G so folds do not
  share drugs. Tuning itself uses `PredefinedSplit` (train → validation).
- **Held-out cases.** 23 interaction rows plus 6 clinical probes are removed before
  splitting and reused in the Brick 11 case study.

---

## 7. Models and tuning

Five classifiers share the same 18 features and training protocol:

| Model | Library | Class handling |
|---|---|---|
| XGBoost | `xgboost` | balanced sample weights, weak-pair weight 0.5 |
| LightGBM | `lightgbm` | balanced sample weights, weak-pair weight 0.5 |
| RandomForest | `scikit-learn` | balanced sample weights, weak-pair weight 0.5 |
| ExtraTrees | `scikit-learn` | balanced sample weights, weak-pair weight 0.5 |
| MLP | `scikit-learn` | standardized inputs, early stopping; no sample weights |

Hyperparameters are tuned with `RandomizedSearchCV` on the validation partition,
scoring macro-F1. The search grids cover depth, learning rate, number of trees,
sampling, regularization and, for MLP, hidden sizes and alpha. The winning model is
refit on the full training partition and frozen to `models/best_model_{mode}.joblib`.

### 7.1 Real held-out results (Mode S, Split G — unseen drugs)

| Model | Accuracy | Macro-F1 | Avoid precision | Avoid recall | Non-sig. recall |
|---|---:|---:|---:|---:|---:|
| **LightGBM** | **0.7525** | **0.5066** | **0.4964** | 0.3203 | 0.8923 |
| MLP | 0.7380 | 0.3879 | 0.4403 | 0.2594 | 0.8933 |
| RandomForest | 0.7002 | 0.4092 | 0.4024 | 0.4641 | 0.7769 |
| XGBoost | 0.5157 | 0.3642 | 0.3042 | 0.7438 | 0.4416 |
| ExtraTrees | 0.4927 | 0.3402 | 0.3089 | 0.8297 | 0.3834 |

Split R (Mode S): LightGBM 0.8036 accuracy / 0.5049 macro-F1, again the best.
Mode P Split G: ExtraTrees 0.3759 macro-F1, LightGBM 0.3614, MLP 0.3657 — all weak,
as expected from the paper-style label leakage. **LightGBM is the deployed model**:
it is the best on both Mode S splits and the most consistent overall.

Class 1 contains only 24 real Mode S pairs, so its F1 is noisy and is not used to
select the model.

---

## 8. Explainability with SHAP

- TreeSHAP is applied to the tuned XGBoost model saved for explanation
  (`models/xgb_final_{mode}.json`), per class, on the test partition.
- Artifacts: `figures/shap_summary_{mode}_{class}.png` and
  `reports/brick10_shap_{mode}.json/md`, six plots in total (three classes × two modes).
- Global summaries show which of the 18 pair descriptors push a prediction toward
  avoid, beneficial or non-significant.
- The about page embeds the Mode S “avoid” summary directly from `/figures/...`.

---

## 9. Calibration and selective prediction

- **Confidence threshold sweep.** At serve time the model answers only when the top
  probability is ≥ 0.60. On Mode S Split G this keeps 88.5% of rows and raises
  accuracy to 78.1%; lower-confidence rows become `uncertain`.
- **Split conformal.** Brick 12 calibrates a conformal predictor at 90% coverage:
  - empirical coverage 0.9213
  - abstention rate 0.4267
  - selective accuracy 0.8628
- **Bootstrap confidence intervals.** Drug-level bootstrap gives a 95% CI for
  macro-F1 of roughly [0.274, 0.446], reflecting the small number of test drugs.
- **Applicability domain.** Morgan fingerprint Tanimoto similarity to the nearest
  training drug; below 0.3 the pair is refused as out of domain.

---

## 10. Inference and serving path

```
POST /api/predict
        │
        ▼
DFIRecommender.recommend(drug, food)
        │
        ├─ 1. normalize names / resolve SMILES / expand food markers
        ├─ 2. clinical rule layer  ──► status "rule"      (curated, conservative)
        ├─ 3. exact-pair evidence  ──► status "evidence"  (labelled dataset row)
        ├─ 4. MOE descriptors + 18 features
        │      └─ LightGBM.predict_proba ──► status "ok" / "uncertain"
        └─ 5. guards: inorganic compound, descriptor failure, domain < 0.3
               └─ status "not_supported"
```

This ordering is deliberate. Pair descriptors are symmetric and cannot identify a
named drug, so established clinical pairs are handled by rules or by their exact
dataset annotation. The ML model is used for chemically plausible pairs that are not
covered by those layers.

Response statuses:

| Status | Meaning |
|---|---|
| `ok` | model scored the pair above the confidence threshold |
| `uncertain` | top probability below 0.6; UI asks for pharmacist advice |
| `rule` | curated clinical rule fired |
| `evidence` | exact labelled drug–compound pair in the dataset |
| `not_supported` | invalid input, inorganic compound, descriptor failure, or out of domain |

---

## 11. Backend architecture (FastAPI)

`web/app.py`:

- Adds `src/` to `sys.path`, imports `DFIRecommender` and name-normalization helpers.
- Keeps one lazily constructed engine per mode in a thread-safe cache; the model is
  loaded once and reused across requests.
- Uses Pydantic request validation (1–300 characters, mode restricted to `S`/`P`).
- Appends label, severity, latency, probability rows and mode metadata to responses.

### Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | landing page (`landing.html`) |
| `/predict` | GET | predictor page (`index.html`) |
| `/about` | GET | project documentation page (`about.html`) |
| `/figures/{name}` | GET | PNG figures used by the about page |
| `/static/*` | GET | CSS, JS and page assets |
| `/api/health` | GET | readiness, drug/compound counts, threshold |
| `/api/examples` | GET | demo pairs for the example chips |
| `/api/suggest` | GET | debounced autocomplete for drug/food inputs |
| `/api/predict` | POST | run the hybrid inference path |

Example request:

```json
POST /api/predict
{"drug": "warfarin", "food": "grape juice", "mode": "S"}
```

Example response (rule layer):

```json
{
  "status": "rule",
  "predicted_class": 0,
  "sentence": "Warfarin should not be taken with food containing grape juice.",
  "reason": "Grape and grapefruit products are flagged with warfarin; keep intake consistent and monitor INR.",
  "label": "Avoid together",
  "severity": "negative",
  "latency_ms": 0.1
}
```

When the ML path answers, the same response additionally carries `confidence` and
`probability_rows` for the three classes.

---

## 12. Frontend architecture

Three static pages, no framework and no build step:

1. **Landing (`/`)** — team name, project title “Food–Drug ML model with SHAP”, two
   buttons (About model → `/about`, Predict → `/predict`), key dataset statistics.
2. **Predictor (`/predict`)** — the existing application: drug/food inputs with
   autocomplete, Specific/Paper-like toggle, example chips, verdict card, probability
   bars, evidence/rule badges, matched compounds, candidates and disclaimer.
3. **About (`/about`)** — abstract, prediction flow, backend/frontend architecture,
   techniques (MOE descriptors, pairwise algebra, Tanimoto/Morgan, feature selection,
   grouped validation, SHAP, conformal), label modes, real metrics table, tech stack,
   limitations and safety, plus embedded SHAP and model-comparison figures.

Implementation notes:

- `app.js` uses `fetch` with `AbortController` so stale autocomplete or prediction
  requests cannot overwrite newer results.
- Rendering is XSS-safe: all dynamic values go through `textContent`, never
  `innerHTML`, except one static inline SVG.
- Accessibility: labelled inputs, `aria-live` result region, `aria-busy` during
  loading, pressed states on the mode toggle, visible focus styles, reduced-motion
  support.
- The UI is responsive down to phone width; probability bars animate on entry.

---

## 13. Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| Data | pandas 2.3, NumPy 2.3, PyArrow 25 |
| Chemistry | RDKit 2026.9, PyBioMed 1.0 (MOE descriptors) |
| ML | scikit-learn 1.7, XGBoost 3.4, LightGBM 4.7 |
| Explainability | SHAP 0.54 (TreeSHAP) |
| Evaluation | GroupKFold, PredefinedSplit, RandomizedSearchCV, split conformal, bootstrap, matplotlib |
| Backend | FastAPI 0.111, Uvicorn 0.29, Pydantic 2.12, joblib |
| Frontend | HTML5, CSS3, vanilla JavaScript (Fetch API), Google Fonts |
| Notebooks | Jupyter, nbformat |
| QA | pytest, `scripts/check_gates.py` |

No database is required. Models, selected features and cleaned tables are local
artifacts; the API reads them at startup.

---

## 14. Artifacts

- Models: `models/best_model_{S,P}.joblib`, `models/xgb_final_{S,P}.json`,
  `models/selected_features_{S,P}.json`, `models/train_drugs_{S,P}.json`,
  `models/food_markers.json`.
- Metrics: `reports/metrics_{S,P}.json`, `reports/model_comparison.*`,
  `reports/cv_results_{S,P}.csv`, `reports/tuning_*_{S,P}.csv`.
- Explainability: `figures/shap_summary_*`, `reports/brick10_shap_*`.
- Cases: `reports/brick11_case_study.json/md`.
- Enhancements: `reports/brick12_enhancements.json/md`.
- Narrative: `reports/report.md`, `reports/retune_audit.md`.

---

## 15. Run and reproduce

```bash
# from dfi/
pip install -r requirements.txt

# full pipeline (long)
bash scripts/run_all.sh

# individual mode
PYTHONPATH=src python src/brick8_models.py --mode S
PYTHONPATH=src python src/brick8b_compare.py --mode S

# web app
python -m uvicorn web.app:app --host 127.0.0.1 --port 8010
# open http://127.0.0.1:8010
```

On Windows, if port 8010 is busy:

```bash
netstat -ano | findstr :8010
taskkill /PID <pid> /F
```

---

## 16. Testing and gates

- `scripts/check_gates.py` checks 31 conditions: data funnels, label counts, feature
  verification, split leakage, model reload equality, evaluation artifacts, SHAP
  figures, case study, enhancements, report and notebooks.
- `tests/` contains API and inference smoke tests.
- `scripts/check_gates.py` currently reports **31/31 gates passed**.
- Manual smoke tests: warfarin + grape juice → avoid (rule), midazolam + grapefruit
  → avoid (rule), warfarin + glucose → abstain.

---

## 17. Known limitations and safety

- The model cannot identify a named drug from pair descriptors alone; clinical
  correctness for known pairs depends on the rule and evidence layers.
- Avoid-class recall on unseen drugs is about 0.32 at the default threshold; the
  system prefers abstention over confident errors.
- Mode S covers 444 labelled drugs; pairs outside that coverage are explicitly marked
  as extrapolation.
- Mode P is for paper comparability only and must not be read as real-world accuracy.
- The external validation set used by the source paper was not obtained; numbers are
  not directly comparable to its 0.9756 accuracy.
- This is a research tool, not a medical device. Confirm every result with a physician
  or pharmacist.
