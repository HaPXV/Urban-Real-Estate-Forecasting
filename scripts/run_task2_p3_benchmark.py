"""
Evidence build script for R2 revision package.
Handles: Subtask E (P3 benchmark), F (H3 vs XGBoost), G (ARIMAX feasibility + Task 1 reassessment)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try: import xgboost as xgb; HAS_XGB=True
except: HAS_XGB=False
try: import lightgbm as lgb; HAS_LGB=True
except: HAS_LGB=False
try: import catboost as cb; HAS_CAT=True
except: HAS_CAT=False
try: import statsmodels.api as sm; HAS_SM=True
except: HAS_SM=False
try: import statsmodels.formula.api as smf; HAS_SMF=True
except: HAS_SMF=False

print(f"XGB={HAS_XGB} LGB={HAS_LGB} CAT={HAS_CAT} SM={HAS_SM}")

OUT = Path("C:/Users/LEGION/DOAN/09_r2_revision/01_evidence_build")
OUT.mkdir(parents=True, exist_ok=True)

ROOT = Path("C:/Users/LEGION/DOAN")

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
df = pd.read_parquet(ROOT / "02_analytic/task2_modeling_trimmed.parquet")
dataset_c = pd.read_parquet(ROOT / "02_analytic/DATASET_C_ANALYTIC_v1.parquet")
geojson_path = ROOT / "11_spatial_outputs_final/spatial_analysis_final.geojson"

print(f"Dataset A trimmed: {df.shape}")
print(f"Dataset C: {dataset_c.shape}, cols: {list(dataset_c.columns)}")

# Add spatial distance feature
import json as _json
with open(geojson_path, encoding='utf-8') as f:
    gj = _json.load(f)

BT_LAT, BT_LON = 10.7723, 106.6983

def haversine(lat1, lon1, lat2=BT_LAT, lon2=BT_LON):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

ward_dist = {}
for feat in gj['features']:
    wk = feat['properties'].get('ward_key')
    geom = feat['geometry']
    if geom['type'] == 'Polygon':
        coords = np.array(geom['coordinates'][0])
    else:
        coords = np.array(geom['coordinates'][0][0])
    cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
    ward_dist[wk] = haversine(cy, cx)

df['dist_ben_thanh_km'] = df['ward_key'].map(ward_dist)
df['log_dist_bt'] = np.log(df['dist_ben_thanh_km'])
df['log_area_m2'] = np.log(df['area_m2'])
print(f"Distance coverage: {df.dist_ben_thanh_km.notna().sum()}/{len(df)}")

# ─────────────────────────────────────────────────────────────
# HELPER: preprocess + metrics
# ─────────────────────────────────────────────────────────────
SEED = 42

def preprocess(df_in, features, target, log_target=True):
    df_p = df_in[features + [target]].copy()
    if 'frontage_m' in df_p.columns:
        df_p['frontage_m'] = df_p['frontage_m'].fillna(df_p['frontage_m'].median())
    cat_cols = [c for c in features if
                df_p[c].dtype == object or
                str(df_p[c].dtype) in ('string','category','str','object') or
                (hasattr(df_p[c].dtype,'kind') and df_p[c].dtype.kind == 'O')]
    for c in cat_cols:
        df_p[c] = df_p[c].astype(str)
    if cat_cols:
        df_p = pd.get_dummies(df_p, columns=cat_cols, drop_first=False, dtype=float)
    X = df_p[[c for c in df_p.columns if c != target]]
    y_raw = df_p[target].values.astype(float)
    y = np.log1p(y_raw) if log_target else y_raw
    return X, y, y_raw

def calc_metrics(y_true, y_pred, label=''):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    r2 = r2_score(y_true, y_pred)
    if label:
        print(f"  {label:<30} R²={r2:.4f}  MAPE={mape:.2f}%  MAE={mae/1e6:.3f}M  RMSE={rmse/1e6:.3f}M")
    return {'RMSE': round(rmse/1e6, 4), 'MAE': round(mae/1e6, 4),
            'MAPE': round(mape, 4), 'R2': round(r2, 4)}

# ─────────────────────────────────────────────────────────────
# SUBTASK E: P3 TEMPORAL SPLIT BENCHMARK
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUBTASK E — P3 TEMPORAL SPLIT BENCHMARK")
print("="*60)

CUTOFF = pd.Timestamp('2025-01-01')
df_sorted = df.sort_values('transaction_date').reset_index(drop=True)
mask_train = df_sorted['transaction_date'] < CUTOFF
mask_test = df_sorted['transaction_date'] >= CUTOFF
print(f"Train (< 2025-01-01): {mask_train.sum()}")
print(f"Test  (>= 2025-01-01): {mask_test.sum()}")

FEATURES_M2 = ['area_m2', 'district', 'ward_key', 'is_mat_tien', 'so_tang',
               'cap_cong_trinh', 'frontage_m', 'transaction_year', 'transaction_quarter']
TARGET_M2 = 'price_m2'

X, y, y_raw = preprocess(df_sorted, FEATURES_M2, TARGET_M2, log_target=True)
X_tr = X[mask_train.values]; X_te = X[mask_test.values]
y_tr = y[mask_train.values]; y_te = y[mask_test.values]
y_te_raw = y_raw[mask_test.values]

print(f"Encoded features: {X.shape[1]}")

p3_results = []

# Location baselines
gm = np.expm1(y_tr.mean())
m = calc_metrics(y_te_raw, np.full(len(y_te_raw), gm), 'GlobalMean')
p3_results.append({'model': 'GlobalMean', 'split': 'P3', 'target': 'price_m2', **m})

# Linear Regression
lr = LinearRegression().fit(X_tr, y_tr)
m = calc_metrics(y_te_raw, np.expm1(lr.predict(X_te)), 'LinearRegression')
p3_results.append({'model': 'LinearRegression', 'split': 'P3', 'target': 'price_m2', **m})

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_tr, y_tr)
m = calc_metrics(y_te_raw, np.expm1(rf.predict(X_te)), 'RandomForest')
p3_results.append({'model': 'RandomForest', 'split': 'P3', 'target': 'price_m2', **m})

if HAS_XGB:
    xm = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                           verbosity=0, n_jobs=-1)
    xm.fit(X_tr, y_tr)
    m = calc_metrics(y_te_raw, np.expm1(xm.predict(X_te)), 'XGBoost')
    p3_results.append({'model': 'XGBoost', 'split': 'P3', 'target': 'price_m2', **m})

if HAS_LGB:
    try:
        lm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                random_state=SEED, n_jobs=-1, verbose=-1)
        lm.fit(X_tr, y_tr)
        m = calc_metrics(y_te_raw, np.expm1(lm.predict(X_te)), 'LightGBM')
        p3_results.append({'model': 'LightGBM', 'split': 'P3', 'target': 'price_m2', **m})
    except Exception as e:
        print(f"  LightGBM FAILED: {e}")

if HAS_CAT:
    cm = cb.CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                               random_seed=SEED, verbose=0)
    cm.fit(X_tr, y_tr)
    m = calc_metrics(y_te_raw, np.expm1(cm.predict(X_te)), 'CatBoost')
    p3_results.append({'model': 'CatBoost', 'split': 'P3', 'target': 'price_m2', **m})

df_p3 = pd.DataFrame(p3_results)
df_p3.to_csv(OUT / 'task2_p3_metrics.csv', index=False)
print(f"\nSaved: task2_p3_metrics.csv")

# ─────────────────────────────────────────────────────────────
# SUBTASK F: H3 AS COMPARABLE HEDONIC + XGBoost vs H3 vs H2
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUBTASK F — H3 vs XGBoost like-for-like comparison")
print("="*60)

if HAS_SM:
    import statsmodels.api as sm

    # Use same P1 random split as existing benchmark (seed=42, 80/20)
    df2 = df.copy()
    df2['log_price_vnd'] = np.log(df2['price_vnd'])
    df2['log_price_m2'] = np.log(df2['price_m2'])
    df2['log_area_m2'] = np.log(df2['area_m2'])
    df2['log_dist_bt'] = np.log(df2['dist_ben_thanh_km'])
    df2['frontage_m_filled'] = df2['frontage_m'].fillna(df2['frontage_m'].median())

    # Ordinal encode cap_cong_trinh
    cap_map = {}
    for i, v in enumerate(sorted(df2['cap_cong_trinh'].dropna().unique())):
        cap_map[v] = i + 1
    df2['cap_ord'] = df2['cap_cong_trinh'].map(cap_map).fillna(1)

    # District dummies
    df2 = pd.get_dummies(df2, columns=['district'], drop_first=True, dtype=float)

    hedonic_cols = ['log_area_m2','is_mat_tien','so_tang','cap_ord',
                    'frontage_m_filled','log_dist_bt','transaction_year']
    dist_cols = [c for c in df2.columns if c.startswith('district_')]
    hedonic_cols += dist_cols

    # P1 split (same as ML benchmark)
    idx = np.arange(len(df2))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42)
    train_h = df2.iloc[idx_tr]
    test_h = df2.iloc[idx_te]

    # H2: target = log(price_vnd)
    X_h2_tr = sm.add_constant(train_h[hedonic_cols])
    X_h2_te = sm.add_constant(test_h[hedonic_cols])
    model_H2 = sm.OLS(train_h['log_price_vnd'], X_h2_tr).fit(cov_type='HC3')

    pred_H2_log = model_H2.predict(X_h2_te)
    pred_H2 = np.exp(pred_H2_log)
    act_H2 = test_h['price_vnd'].values

    mae_H2 = mean_absolute_error(act_H2, pred_H2)
    rmse_H2 = np.sqrt(mean_squared_error(act_H2, pred_H2))
    mape_H2 = np.mean(np.abs((act_H2 - pred_H2)/np.clip(np.abs(act_H2),1e-6,None)))*100
    r2_H2 = r2_score(act_H2, pred_H2)
    r2_H2_is = model_H2.rsquared
    r2_H2_adj = model_H2.rsquared_adj
    print(f"  H2 (log_price_vnd): in-sample R²={r2_H2_is:.4f} AdjR²={r2_H2_adj:.4f}")
    print(f"  H2 test: R²={r2_H2:.4f}  MAPE={mape_H2:.2f}%  MAE={mae_H2/1e9:.4f}B VND")

    # H3: target = log(price_m2)
    X_h3_tr = sm.add_constant(train_h[hedonic_cols])
    X_h3_te = sm.add_constant(test_h[hedonic_cols])
    model_H3 = sm.OLS(train_h['log_price_m2'], X_h3_tr).fit(cov_type='HC3')

    pred_H3_log = model_H3.predict(X_h3_te)
    pred_H3 = np.exp(pred_H3_log)
    act_H3 = test_h['price_m2'].values

    mae_H3 = mean_absolute_error(act_H3, pred_H3)
    rmse_H3 = np.sqrt(mean_squared_error(act_H3, pred_H3))
    mape_H3 = np.mean(np.abs((act_H3 - pred_H3)/np.clip(np.abs(act_H3),1e-6,None)))*100
    r2_H3 = r2_score(act_H3, pred_H3)
    r2_H3_is = model_H3.rsquared
    r2_H3_adj = model_H3.rsquared_adj
    print(f"  H3 (log_price_m2):  in-sample R²={r2_H3_is:.4f} AdjR²={r2_H3_adj:.4f}")
    print(f"  H3 test: R²={r2_H3:.4f}  MAPE={mape_H3:.2f}%  MAE={mae_H3/1e6:.3f}M VND/m²")

    # H3 coefficients
    h3_coefs = pd.DataFrame({
        'variable': model_H3.params.index,
        'coef': model_H3.params.values,
        'se_hc3': model_H3.bse.values,
        't': model_H3.tvalues.values,
        'p': model_H3.pvalues.values,
    })
    h3_coefs['sig'] = h3_coefs['p'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ('.' if p < 0.1 else ''))))
    h3_coefs.to_csv(OUT / 'h3_metrics.csv', index=False)

    # XGBoost P1 results (from existing verified outputs)
    xgb_p1 = {'model': 'XGBoost', 'target': 'price_m2', 'split': 'P1',
               'R2_insample': 'N/A (ML no in-sample R² reported)',
               'R2_test': 0.681, 'MAPE_test': 16.78, 'MAE_test_scale': '16.11 M VND/m²'}

    compare_rows = [
        {'model': 'XGBoost', 'target': 'price_m2', 'split': 'P1_random',
         'R2_insample': 'N/A', 'R2_test': 0.681,
         'MAPE_test_pct': 16.78, 'MAE_test': '16.11 M VND/m²', 'note': 'Primary ML benchmark'},
        {'model': 'H3 (OLS HC3)', 'target': 'log(price_m2)', 'split': 'P1_random',
         'R2_insample': round(r2_H3_is, 4), 'R2_test': round(r2_H3, 4),
         'MAPE_test_pct': round(mape_H3, 2), 'MAE_test': f'{mae_H3/1e6:.3f} M VND/m²',
         'note': 'Like-for-like hedonic comparison (same price_m2 concept)'},
        {'model': 'H2 (OLS HC3)', 'target': 'log(price_vnd)', 'split': 'P1_random',
         'R2_insample': round(r2_H2_is, 4), 'R2_test': round(r2_H2, 4),
         'MAPE_test_pct': round(mape_H2, 2), 'MAE_test': f'{mae_H2/1e9:.4f} B VND (total price)',
         'note': 'Economic interpretation model — different target scale'},
    ]
    pd.DataFrame(compare_rows).to_csv(OUT / 'xgb_h3_h2_compare.csv', index=False)
    print("Saved: h3_metrics.csv, xgb_h3_h2_compare.csv")
else:
    print("statsmodels not available — skipping H3 re-run")

# ─────────────────────────────────────────────────────────────
# SUBTASK G: TASK 1 ARIMAX FEASIBILITY + REASSESSMENT
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUBTASK G — Task 1 Reassessment + ARIMAX feasibility")
print("="*60)

# Build Task 1 monthly series from the FULL dataset (5171 records — but we have 5005+155)
# The trimmed dataset was used for Task 1 series in the paper's construction
# Use trimmed for consistency, check transaction_date range
df_task1 = df.copy()
df_task1['ym'] = df_task1['transaction_date'].dt.to_period('M')
monthly = df_task1.groupby('ym')['price_m2_million'].mean().reset_index()
monthly.columns = ['year_month', 'avg_price_m2_million']
monthly = monthly.sort_values('year_month').reset_index(drop=True)
print(f"Monthly series: {len(monthly)} observations")
print(f"Range: {monthly.year_month.min()} to {monthly.year_month.max()}")

# Align with Dataset C
dataset_c['ym'] = dataset_c['year'].astype(str) + '-' + dataset_c['month'].astype(str).str.zfill(2)
dataset_c['ym_period'] = pd.PeriodIndex(dataset_c['ym'], freq='M')

# Merge
monthly['ym_period'] = monthly['year_month']
merged = monthly.merge(dataset_c.rename(columns={'ym_period':'ym_period'}),
                       on='ym_period', how='left')
print(f"Merged shape: {merged.shape}")
print(f"Dataset C coverage: {merged.interest_rate.notna().sum()}/{len(merged)}")
print(f"Dataset C cols: {[c for c in dataset_c.columns if c not in ['year','month','ym','ym_period']]}")

# TRAIN/TEST split for Task 1: 37 train / 10 test
TRAIN_N = 37
series_full = merged['avg_price_m2_million'].values
series_train = series_full[:TRAIN_N]
series_test = series_full[TRAIN_N:]
print(f"Train series: {len(series_train)} obs, Test: {len(series_test)} obs")

# Exogenous vars for ARIMAX
exog_cols = ['interest_rate','gold_price_sjc','vn_index','cpi_index']
exog_all = merged[exog_cols].values
exog_train = exog_all[:TRAIN_N]
exog_test = exog_all[TRAIN_N:]

# Check for missing exog
print(f"Exog train NaN: {np.isnan(exog_train).sum()}")
print(f"Exog test  NaN: {np.isnan(exog_test).sum()}")

t1_results = []

def t1_metrics(actual, pred, label):
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred)**2))
    mape = np.mean(np.abs((actual - pred)/np.clip(np.abs(actual),1e-8,None)))*100
    print(f"  {label:<20} MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.4f}%")
    return {'model': label, 'MAE': round(mae,4), 'RMSE': round(rmse,4), 'MAPE': round(mape,4)}

# Naive
naive_pred = np.full(len(series_test), series_train[-1])
t1_results.append(t1_metrics(series_test, naive_pred, 'Naive'))

# MA(3)
ma3_pred = np.full(len(series_test), np.mean(series_train[-3:]))
t1_results.append(t1_metrics(series_test, ma3_pred, 'MA(3)'))

# Verified results from config.py (already established)
# Prophet and ARIMA known: add as verified rows
t1_results.append({'model':'ARIMA(3,1,3)','MAE':7.68,'RMSE':9.57,'MAPE':7.32,'source':'verified_config'})
t1_results.append({'model':'Prophet','MAE':6.49,'RMSE':9.02,'MAPE':6.46,'source':'verified_config'})
t1_results.append({'model':'LSTM','MAE':8.86,'RMSE':10.78,'MAPE':8.46,'source':'verified_config'})

# ARIMAX attempt
arimax_feasible = False
arimax_error = None
if HAS_SM:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # Standardize exog to avoid scale issues
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    exog_tr_sc = scaler.fit_transform(exog_train)
    exog_te_sc = scaler.transform(exog_test)
    try:
        model_arimax = SARIMAX(series_train, exog=exog_tr_sc,
                               order=(3,1,3), enforce_stationarity=False,
                               enforce_invertibility=False)
        res_arimax = model_arimax.fit(disp=False, maxiter=200)
        pred_arimax = res_arimax.forecast(steps=len(series_test), exog=exog_te_sc)
        m = t1_metrics(series_test, pred_arimax, 'ARIMAX(3,1,3)')
        m['source'] = 'computed'
        t1_results.append(m)
        arimax_feasible = True
        print(f"  ARIMAX AIC={res_arimax.aic:.2f}  BIC={res_arimax.bic:.2f}")
        print(f"  ARIMAX exog sig (p<0.1): {[exog_cols[i] for i,p in enumerate(res_arimax.pvalues[3:7]) if p < 0.1]}")
        # Save ARIMAX metrics
        pd.DataFrame([m]).to_csv(OUT / 'task1_arimax_metrics.csv', index=False)
    except Exception as e:
        arimax_error = str(e)
        print(f"  ARIMAX FAILED: {e}")

df_t1 = pd.DataFrame(t1_results)
df_t1.to_csv(OUT / 'task1_reassessment_metrics.csv', index=False)

print(f"\nARIMAX feasible: {arimax_feasible}")
if arimax_error:
    print(f"ARIMAX error: {arimax_error}")

# Save ARIMAX feasibility result for later
with open(OUT / 'arimax_status.json', 'w') as f:
    json.dump({'feasible': arimax_feasible, 'error': arimax_error,
               'exog_cols': exog_cols, 'train_n': TRAIN_N, 'test_n': len(series_test)}, f)

print("\n=== ALL COMPUTATIONS DONE ===")
print(f"Outputs in: {OUT}")
