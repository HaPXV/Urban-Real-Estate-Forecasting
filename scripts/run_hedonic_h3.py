"""Subtask F: H3 hedonic re-run and comparison with H2 and XGBoost."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import json

OUT = Path("C:/Users/LEGION/DOAN/09_r2_revision/01_evidence_build")
ROOT = Path("C:/Users/LEGION/DOAN")

df = pd.read_parquet(ROOT / "02_analytic/task2_modeling_trimmed.parquet")

# Build spatial distance
import json as _json
with open(ROOT / "11_spatial_outputs_final/spatial_analysis_final.geojson", encoding='utf-8') as f:
    gj = _json.load(f)
BT_LAT, BT_LON = 10.7723, 106.6983
def haversine(lat1, lon1):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(BT_LAT)
    dphi = np.radians(BT_LAT - lat1); dlam = np.radians(BT_LON - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

ward_dist = {}
for feat in gj['features']:
    wk = feat['properties'].get('ward_key')
    geom = feat['geometry']
    coords = np.array(geom['coordinates'][0] if geom['type']=='Polygon' else geom['coordinates'][0][0])
    cx, cy = coords[:,0].mean(), coords[:,1].mean()
    ward_dist[wk] = haversine(cy, cx)

df['dist_ben_thanh_km'] = df['ward_key'].map(ward_dist)

# Feature engineering
df['log_price_vnd'] = np.log(df['price_vnd'].astype(float))
df['log_price_m2']  = np.log(df['price_m2'].astype(float))
df['log_area_m2']   = np.log(df['area_m2'].astype(float))
df['log_dist_bt']   = np.log(df['dist_ben_thanh_km'].astype(float))
df['frontage_m_filled'] = df['frontage_m'].fillna(df['frontage_m'].median()).astype(float)
df['is_mat_tien']   = df['is_mat_tien'].astype(float)
df['so_tang']       = df['so_tang'].astype(float)
df['transaction_year'] = df['transaction_year'].astype(float)

# Ordinal cap
cap_map = {v: i+1 for i, v in enumerate(sorted(df['cap_cong_trinh'].dropna().unique()))}
df['cap_ord'] = df['cap_cong_trinh'].map(cap_map).fillna(1.0).astype(float)

# District dummies — force float
dist_dummies = pd.get_dummies(df['district'], drop_first=True, dtype=float)
dist_cols = list(dist_dummies.columns)
df = pd.concat([df, dist_dummies], axis=1)

HEDONIC_COLS = ['log_area_m2','is_mat_tien','so_tang','cap_ord',
                'frontage_m_filled','log_dist_bt','transaction_year'] + dist_cols

# Ensure all float
for c in HEDONIC_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
df = df.dropna(subset=HEDONIC_COLS + ['log_price_vnd','log_price_m2'])

# P1 split (same as ML: random 80/20, seed=42)
idx = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42)
train_h = df.iloc[idx_tr]
test_h  = df.iloc[idx_te]
print(f"Train: {len(train_h)}  Test: {len(test_h)}")

def run_ols(train, test, target, label):
    X_tr = sm.add_constant(train[HEDONIC_COLS].astype(float), has_constant='add')
    X_te = sm.add_constant(test[HEDONIC_COLS].astype(float),  has_constant='add')
    # Align columns
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
    model = sm.OLS(train[target].astype(float), X_tr.astype(float)).fit(cov_type='HC3')
    pred_log = model.predict(X_te.astype(float))
    pred = np.exp(pred_log)
    actual = np.exp(test[target].astype(float).values)
    mae  = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred)/np.clip(np.abs(actual),1e-6,None)))*100
    r2   = r2_score(actual, pred)
    print(f"  {label}: in-sample R²={model.rsquared:.4f} AdjR²={model.rsquared_adj:.4f}")
    print(f"    test R²={r2:.4f}  MAPE={mape:.2f}%  MAE={mae:.2f}")
    return model, pred, actual, {
        'R2_insample': round(model.rsquared, 4),
        'AdjR2_insample': round(model.rsquared_adj, 4),
        'R2_test': round(r2, 4), 'MAPE_test': round(mape, 4),
        'MAE_raw': round(mae, 2), 'RMSE_raw': round(rmse, 2)
    }

print("\n--- H2: target = log(price_vnd) ---")
m_H2, _, _, stats_H2 = run_ols(train_h, test_h, 'log_price_vnd', 'H2')

print("\n--- H3: target = log(price_m2) ---")
m_H3, pred_H3, act_H3, stats_H3 = run_ols(train_h, test_h, 'log_price_m2', 'H3')

# Save H3 coefficients
h3_coefs = pd.DataFrame({
    'variable': m_H3.params.index,
    'coef': m_H3.params.values,
    'se_hc3': m_H3.bse.values,
    't': m_H3.tvalues.values,
    'p': m_H3.pvalues.values,
})
h3_coefs['sig'] = h3_coefs['p'].apply(
    lambda p: '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ('.' if p<0.1 else ''))))
h3_coefs.to_csv(OUT / 'h3_coefficients.csv', index=False)
print("\nH3 key coefficients:")
print(h3_coefs.to_string(index=False))

# XGB P1 verified
compare = [
    {'model':'XGBoost (M2 extended)', 'target':'price_m2 (log1p)', 'split':'P1_random_80-20',
     'R2_insample':'N/A','R2_test':0.681,'MAPE_test_pct':16.78,
     'MAE_test':'16.11 M VND/m²','comparable_to_h3':'YES (same price concept)'},
    {'model':'H3 (OLS HC3)', 'target':'log(price_m2)', 'split':'P1_random_80-20',
     'R2_insample':stats_H3['R2_insample'],'R2_test':stats_H3['R2_test'],
     'MAPE_test_pct':stats_H3['MAPE_test'],
     'MAE_test':f"{stats_H3['MAE_raw']/1e6:.3f} M VND/m²",
     'comparable_to_h3':'YES — like-for-like (same price_m2 target concept)'},
    {'model':'H2 (OLS HC3)', 'target':'log(price_vnd)', 'split':'P1_random_80-20',
     'R2_insample':stats_H2['R2_insample'],'R2_test':stats_H2['R2_test'],
     'MAPE_test_pct':stats_H2['MAPE_test'],
     'MAE_test':f"{stats_H2['MAE_raw']/1e9:.4f} B VND (total price)",
     'comparable_to_h3':'NO — different target (total price), use for economic coefs only'},
]
pd.DataFrame(compare).to_csv(OUT / 'xgb_h3_h2_compare.csv', index=False)

# Summary metrics
pd.DataFrame([
    {'model':'H2','target':'log_price_vnd', **stats_H2},
    {'model':'H3','target':'log_price_m2',  **stats_H3},
]).to_csv(OUT / 'h3_metrics.csv', index=False)

print(f"\nSaved: h3_coefficients.csv, h3_metrics.csv, xgb_h3_h2_compare.csv")
print(f"\nH3 MAPE={stats_H3['MAPE_test']:.2f}%  R²={stats_H3['R2_test']:.4f}")
print(f"H2 MAPE={stats_H2['MAPE_test']:.2f}%  R²={stats_H2['R2_test']:.4f}")
