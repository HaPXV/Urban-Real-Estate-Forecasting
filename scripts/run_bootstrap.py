"""
Paired bootstrap inference: XGBoost vs H3 on P3 test set.
Reproduces the exact P3 split used in the evidence build, trains both models,
verifies MAPE matches stored values, then runs 5000-replicate bootstrap.

Stored reference values (from corrected_p3_metrics.csv + h3_split_decision.md):
  XGBoost P3 MAPE = 16.5465%
  H3      P3 MAPE = 18.2659%
  Gap (H3 - XGBoost) = 1.7194 pp
  N_test = 1658
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import xgboost as xgb

SEED = 42
ROOT = Path("C:/Users/LEGION/DOAN")
OUT  = Path("C:/Users/LEGION/DOAN/09_r2_revision/03_bootstrap")
OUT.mkdir(parents=True, exist_ok=True)

# ── Reference locked values ────────────────────────────────────────────────
REF_XGB_MAPE  = 16.5465   # %
REF_H3_MAPE   = 18.2659   # %
REF_XGB_MAE   = 18.0665   # M VND/m²
REF_H3_MAE    = 19.9405   # M VND/m²
TOLERANCE_PP  = 0.05       # allowable MAPE delta vs stored value

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_parquet(ROOT / "02_analytic/task2_modeling_trimmed.parquet")
print(f"Dataset: {df.shape}")

# Spatial feature (haversine to Ben Thanh market)
import json as _json
with open(ROOT / "11_spatial_outputs_final/spatial_analysis_final.geojson", encoding='utf-8') as f:
    gj = _json.load(f)
BT_LAT, BT_LON = 10.7723, 106.6983

def haversine(lat1, lon1):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(BT_LAT)
    dphi = np.radians(BT_LAT - lat1)
    dlam = np.radians(BT_LON - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

ward_dist = {}
for feat in gj['features']:
    wk   = feat['properties'].get('ward_key')
    geom = feat['geometry']
    coords = np.array(geom['coordinates'][0] if geom['type']=='Polygon' else geom['coordinates'][0][0])
    cx, cy = coords[:,0].mean(), coords[:,1].mean()
    ward_dist[wk] = haversine(cy, cx)

df['dist_ben_thanh_km'] = df['ward_key'].map(ward_dist)
df['log_dist_bt']       = np.log(df['dist_ben_thanh_km'])
df['log_area_m2']       = np.log(df['area_m2'])
print(f"Distance coverage: {df.dist_ben_thanh_km.notna().sum()}/{len(df)}")

# ── P3 temporal split (CUTOFF = 2025-01-01) ───────────────────────────────
CUTOFF    = pd.Timestamp('2025-01-01')
df_sorted = df.sort_values('transaction_date').reset_index(drop=True)
mask_train = df_sorted['transaction_date'] < CUTOFF
mask_test  = df_sorted['transaction_date'] >= CUTOFF

N_TRAIN = int(mask_train.sum())
N_TEST  = int(mask_test.sum())
print(f"P3 train: {N_TRAIN}  test: {N_TEST}")

# ══════════════════════════════════════════════════════════════════
# TASK A — XGBoost P3 predictions
# ══════════════════════════════════════════════════════════════════
print("\n=== XGBoost P3 ===")
FEATURES_M2 = ['area_m2', 'district', 'ward_key', 'is_mat_tien', 'so_tang',
               'cap_cong_trinh', 'frontage_m', 'transaction_year', 'transaction_quarter']
TARGET_M2 = 'price_m2'

def preprocess_ml(df_in, features, target):
    df_p = df_in[features + [target]].copy()
    if 'frontage_m' in df_p.columns:
        df_p['frontage_m'] = df_p['frontage_m'].fillna(df_p['frontage_m'].median())
    # Detect non-numeric columns robustly
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(df_p[c])]
    for c in cat_cols:
        df_p[c] = df_p[c].astype(str)
    if cat_cols:
        df_p = pd.get_dummies(df_p, columns=cat_cols, drop_first=False, dtype=float)
    # Ensure all columns are numeric
    X = df_p[[c for c in df_p.columns if c != target]].astype(float)
    y_raw = df_p[target].values.astype(float)
    y_log = np.log1p(y_raw)
    return X, y_log, y_raw

X_all, y_log, y_raw_all = preprocess_ml(df_sorted, FEATURES_M2, TARGET_M2)
X_tr_xgb = X_all[mask_train.values]
X_te_xgb = X_all[mask_test.values]
y_tr_xgb = y_log[mask_train.values]
y_te_raw  = y_raw_all[mask_test.values]  # ground truth (raw price_m2)

xgm = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                        verbosity=0, n_jobs=-1)
xgm.fit(X_tr_xgb, y_tr_xgb)
pred_xgb = np.expm1(xgm.predict(X_te_xgb))

mape_xgb = np.mean(np.abs((y_te_raw - pred_xgb) / np.clip(np.abs(y_te_raw), 1e-6, None))) * 100
mae_xgb  = mean_absolute_error(y_te_raw, pred_xgb)
r2_xgb   = r2_score(y_te_raw, pred_xgb)
print(f"  XGBoost MAPE={mape_xgb:.4f}%  MAE={mae_xgb/1e6:.4f}M  R²={r2_xgb:.4f}")
print(f"  Reference:   MAPE={REF_XGB_MAPE:.4f}%  MAE={REF_XGB_MAE:.4f}M")
xgb_mape_ok = abs(mape_xgb - REF_XGB_MAPE) <= TOLERANCE_PP
print(f"  Match: {'✓ OK' if xgb_mape_ok else '✗ MISMATCH (delta=' + str(round(abs(mape_xgb-REF_XGB_MAPE),4)) + ' pp)'}")

# ══════════════════════════════════════════════════════════════════
# TASK B — H3 P3 predictions (OLS HC3, same feature set)
# ══════════════════════════════════════════════════════════════════
print("\n=== H3 OLS (HC3) P3 ===")

df2 = df_sorted.copy()
df2['log_price_m2']       = np.log(df2['price_m2'].astype(float))
df2['log_area_m2']        = np.log(df2['area_m2'].astype(float))
df2['log_dist_bt']        = np.log(df2['dist_ben_thanh_km'].astype(float))
df2['frontage_m_filled']  = df2['frontage_m'].fillna(df2['frontage_m'].median()).astype(float)
df2['is_mat_tien']        = df2['is_mat_tien'].astype(float)
df2['so_tang']            = df2['so_tang'].astype(float)
df2['transaction_year']   = df2['transaction_year'].astype(float)

cap_map = {v: i+1 for i, v in enumerate(sorted(df2['cap_cong_trinh'].dropna().unique()))}
df2['cap_ord'] = df2['cap_cong_trinh'].map(cap_map).fillna(1.0).astype(float)

dist_dummies = pd.get_dummies(df2['district'], drop_first=True, dtype=float)
dist_cols    = list(dist_dummies.columns)
df2          = pd.concat([df2.reset_index(drop=True), dist_dummies.reset_index(drop=True)], axis=1)

HEDONIC_COLS = ['log_area_m2', 'is_mat_tien', 'so_tang', 'cap_ord',
                'frontage_m_filled', 'log_dist_bt', 'transaction_year'] + dist_cols

for c in HEDONIC_COLS:
    df2[c] = pd.to_numeric(df2[c], errors='coerce').astype(float)
df2 = df2.dropna(subset=HEDONIC_COLS + ['log_price_m2'])

# Re-create masks on cleaned df2 (index-aligned after dropna)
mask_train_h3 = df2['transaction_date'] < CUTOFF
mask_test_h3  = df2['transaction_date'] >= CUTOFF

train_h3 = df2[mask_train_h3]
test_h3  = df2[mask_test_h3]
print(f"  H3 train: {len(train_h3)}  test: {len(test_h3)}")

X_h3_tr = sm.add_constant(train_h3[HEDONIC_COLS].astype(float), has_constant='add')
X_h3_te = sm.add_constant(test_h3[HEDONIC_COLS].astype(float),  has_constant='add')
X_h3_te = X_h3_te.reindex(columns=X_h3_tr.columns, fill_value=0.0)

model_H3   = sm.OLS(train_h3['log_price_m2'].astype(float), X_h3_tr.astype(float)).fit(cov_type='HC3')
pred_H3_log = model_H3.predict(X_h3_te.astype(float))
pred_h3     = np.exp(pred_H3_log.values)
act_h3      = np.exp(test_h3['log_price_m2'].values)

mape_h3 = np.mean(np.abs((act_h3 - pred_h3) / np.clip(np.abs(act_h3), 1e-6, None))) * 100
mae_h3  = mean_absolute_error(act_h3, pred_h3)
r2_h3   = r2_score(act_h3, pred_h3)
print(f"  H3 MAPE={mape_h3:.4f}%  MAE={mae_h3/1e6:.4f}M  R²={r2_h3:.4f}")
print(f"  Reference: MAPE={REF_H3_MAPE:.4f}%  MAE={REF_H3_MAE:.4f}M")
h3_mape_ok = abs(mape_h3 - REF_H3_MAPE) <= TOLERANCE_PP
print(f"  Match: {'✓ OK' if h3_mape_ok else '✗ MISMATCH (delta=' + str(round(abs(mape_h3-REF_H3_MAPE),4)) + ' pp)'}")

# ══════════════════════════════════════════════════════════════════
# TASK C — Alignment check
# ══════════════════════════════════════════════════════════════════
print("\n=== Alignment check ===")
# XGBoost prediction vector length
n_xgb = len(pred_xgb)
# H3 prediction vector length
n_h3  = len(pred_h3)

print(f"  XGBoost test observations: {n_xgb}")
print(f"  H3      test observations: {n_h3}")
print(f"  Reference N_test: 1658")

# Check alignment: both must be exactly the P3 test set
# XGBoost uses mask_test on df_sorted (no dropna — all features filled)
# H3 uses mask_test_h3 on df2 (after dropna on HEDONIC_COLS)
# If N differs, we take the intersection
if n_xgb == n_h3:
    print("  ✓ Same test size — using direct vector alignment")
    y_true_xgb = y_te_raw
    y_true_h3  = act_h3
    # Verify same underlying actuals (within floating-point)
    # XGBoost actuals are price_m2 from df_sorted; H3 actuals are exp(log_price_m2) = price_m2 from df2
    # Both are sorted by transaction_date, so should be identical
    match_actuals = np.allclose(y_te_raw, act_h3, rtol=1e-4)
    print(f"  Actuals match: {'✓ YES' if match_actuals else '✗ MISMATCH'}")
    ALIGNED = match_actuals
else:
    print(f"  ✗ Test size mismatch: XGBoost={n_xgb} vs H3={n_h3}")
    print("  Attempting index-based intersection...")
    # Get original indices for test set in each pipeline
    xgb_test_idx = df_sorted.index[mask_test.values].tolist()
    h3_test_idx  = df2.index[mask_test_h3.values].tolist()
    common_idx   = sorted(set(xgb_test_idx) & set(h3_test_idx))
    print(f"  Intersection size: {len(common_idx)}")
    if len(common_idx) < 100:
        print("  ABORT: intersection too small for valid comparison")
        ALIGNED = False
    else:
        # Map back to prediction arrays
        xgb_pos = [xgb_test_idx.index(i) for i in common_idx]
        h3_pos  = [h3_test_idx.index(i)  for i in common_idx]
        pred_xgb_al = pred_xgb[xgb_pos]
        pred_h3_al  = pred_h3[h3_pos]
        y_true_xgb  = y_te_raw[xgb_pos]
        y_true_h3   = act_h3[h3_pos]
        print(f"  Aligned on {len(common_idx)} common test observations")
        ALIGNED = True

if not ALIGNED:
    print("\nCOMPARISON NOT VALID — Stopping.")
    result = {"valid": False, "reason": "Test set alignment failed"}
    with open(OUT / 'bootstrap_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    sys.exit(1)

# Recompute MAPE on aligned vectors (in case intersection was used)
if n_xgb == n_h3:
    pred_xgb_al = pred_xgb
    pred_h3_al  = pred_h3

ape_xgb = np.abs((y_true_xgb - pred_xgb_al) / np.clip(np.abs(y_true_xgb), 1e-6, None)) * 100
ape_h3  = np.abs((y_true_h3  - pred_h3_al)  / np.clip(np.abs(y_true_h3),  1e-6, None)) * 100
ae_xgb  = np.abs(y_true_xgb - pred_xgb_al)
ae_h3   = np.abs(y_true_h3  - pred_h3_al)

obs_mape_xgb = ape_xgb.mean()
obs_mape_h3  = ape_h3.mean()
obs_mae_xgb  = ae_xgb.mean()
obs_mae_h3   = ae_h3.mean()

obs_mape_gap = obs_mape_h3 - obs_mape_xgb    # positive = XGBoost better
obs_mae_gap  = obs_mae_h3  - obs_mae_xgb

print(f"\n  Aligned test N: {len(ape_xgb)}")
print(f"  XGBoost MAPE: {obs_mape_xgb:.4f}%   MAE: {obs_mae_xgb/1e6:.4f} M")
print(f"  H3      MAPE: {obs_mape_h3:.4f}%   MAE: {obs_mae_h3/1e6:.4f} M")
print(f"  Gap (H3-XGBoost) MAPE: {obs_mape_gap:.4f} pp")
print(f"  Gap (H3-XGBoost) MAE:  {obs_mae_gap/1e6:.4f} M")

# ══════════════════════════════════════════════════════════════════
# TASK D — Paired bootstrap (5000 replicates)
# ══════════════════════════════════════════════════════════════════
print("\n=== Paired bootstrap (B=5000) ===")
N_BOOT = 5000
rng    = np.random.default_rng(seed=SEED)
N      = len(ape_xgb)

boot_mape_gap = np.empty(N_BOOT)
boot_mae_gap  = np.empty(N_BOOT)

for b in range(N_BOOT):
    idx = rng.integers(0, N, size=N)   # sample with replacement
    boot_mape_gap[b] = ape_h3[idx].mean()  - ape_xgb[idx].mean()
    boot_mae_gap[b]  = ae_h3[idx].mean()   - ae_xgb[idx].mean()

# 95% percentile CI
mape_ci_lo, mape_ci_hi = np.percentile(boot_mape_gap, [2.5, 97.5])
mae_ci_lo,  mae_ci_hi  = np.percentile(boot_mae_gap,  [2.5, 97.5])

# One-sided p-value: P(gap > 0) = fraction of bootstrap replicates where H3 MAPE > XGBoost MAPE
p_one_sided_mape = (boot_mape_gap <= 0).mean()   # H0: gap <= 0 (XGBoost not better)
p_one_sided_mae  = (boot_mae_gap  <= 0).mean()

print(f"  MAPE gap observed:      {obs_mape_gap:.4f} pp")
print(f"  MAPE gap bootstrap mean:{boot_mape_gap.mean():.4f} pp")
print(f"  MAPE gap 95% CI:        [{mape_ci_lo:.4f}, {mape_ci_hi:.4f}] pp")
print(f"  P-value (one-sided):    {p_one_sided_mape:.4f}")
print()
print(f"  MAE  gap observed:      {obs_mae_gap/1e6:.4f} M VND/m²")
print(f"  MAE  gap bootstrap mean:{boot_mae_gap.mean()/1e6:.4f} M VND/m²")
print(f"  MAE  gap 95% CI:        [{mae_ci_lo/1e6:.4f}, {mae_ci_hi/1e6:.4f}] M VND/m²")
print(f"  P-value (one-sided):    {p_one_sided_mae:.4f}")

# ── Decision: CI crosses zero? ─────────────────────────────────────
mape_ci_positive = (mape_ci_lo > 0) and (mape_ci_hi > 0)
mae_ci_positive  = (mae_ci_lo  > 0) and (mae_ci_hi  > 0)
print(f"\n  CI entirely above 0 (MAPE): {'YES' if mape_ci_positive else 'NO — CI crosses zero'}")
print(f"  CI entirely above 0 (MAE):  {'YES' if mae_ci_positive  else 'NO — CI crosses zero'}")

# ══════════════════════════════════════════════════════════════════
# TASK E — Save results
# ══════════════════════════════════════════════════════════════════
result = {
    "valid": True,
    "alignment": {
        "n_test": N,
        "xgb_model_ok": bool(xgb_mape_ok),
        "h3_model_ok":  bool(h3_mape_ok),
        "actuals_match": bool(match_actuals) if n_xgb == n_h3 else "intersected"
    },
    "observed": {
        "mape_xgb_pct": round(obs_mape_xgb, 4),
        "mape_h3_pct":  round(obs_mape_h3,  4),
        "mape_gap_pp":  round(obs_mape_gap,  4),
        "mae_xgb_M":    round(obs_mae_xgb/1e6, 4),
        "mae_h3_M":     round(obs_mae_h3/1e6,  4),
        "mae_gap_M":    round(obs_mae_gap/1e6,  4),
    },
    "bootstrap": {
        "n_replicates": N_BOOT,
        "seed": SEED,
        "mape_gap_boot_mean_pp":   round(boot_mape_gap.mean(), 4),
        "mape_gap_ci_lo_pp":       round(mape_ci_lo, 4),
        "mape_gap_ci_hi_pp":       round(mape_ci_hi, 4),
        "mape_p_one_sided":        round(p_one_sided_mape, 4),
        "mape_ci_entirely_positive": bool(mape_ci_positive),
        "mae_gap_boot_mean_M":     round(boot_mae_gap.mean()/1e6, 4),
        "mae_gap_ci_lo_M":         round(mae_ci_lo/1e6, 4),
        "mae_gap_ci_hi_M":         round(mae_ci_hi/1e6, 4),
        "mae_p_one_sided":         round(p_one_sided_mae, 4),
        "mae_ci_entirely_positive": bool(mae_ci_positive),
    },
    "reference_locked": {
        "xgb_mape_ref": REF_XGB_MAPE,
        "h3_mape_ref":  REF_H3_MAPE,
        "gap_ref":      round(REF_H3_MAPE - REF_XGB_MAPE, 4)
    }
}

with open(OUT / 'bootstrap_results.json', 'w') as f:
    json.dump(result, f, indent=2)

# Save per-observation errors for reproducibility audit
pd.DataFrame({
    'ape_xgb': ape_xgb,
    'ape_h3':  ape_h3,
    'ae_xgb':  ae_xgb / 1e6,
    'ae_h3':   ae_h3  / 1e6,
    'diff_ape': ape_h3 - ape_xgb,
}).to_csv(OUT / 'bootstrap_obs_errors.csv', index=False)

print(f"\nSaved: bootstrap_results.json, bootstrap_obs_errors.csv")
print("=== DONE ===")
