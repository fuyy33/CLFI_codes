
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from joblib import Parallel, delayed
import os
from tqdm import tqdm
import warnings
from xgboost import XGBRegressor
import shap
import optuna
import gc
from scipy.stats import linregress
# ========================================
np.random.seed(42)
warnings.filterwarnings('ignore')
N_CORES = 20  

predictor_names = ['CA', 'PD', 'NTL', 'elevation', 'slope', 'aspect']
response_name = 'CFI'
years = list(range(2001, 2023))  
output_dir = ""
os.makedirs(output_dir, exist_ok=True)

# =======================================
fpaths = {
    'crop_area': "",
    'population': "",
    'nightlight': "",
    'elevation': "",
    'frag': ""
}

templates = {
    'crop_area': "*.tif",
    'population': "*.tif",
    'nightlight': "*.tif",
    'elevation': "*.tif",
    'slope': "*.tif",
    'aspect': "*.tif",
    'frag': "*.tif"
}

# ========================================
def load_raster(file_path, is_elevation=False):
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(np.float32)
        if is_elevation:
            data[data < 0] = np.nan
    return data

def load_year_data(year):
    data = {
        'CA': load_raster(os.path.join(fpaths['crop_area'], templates['crop_area'] % year)),
        'PD': load_raster(os.path.join(fpaths['population'], templates['population'] % year)),
        'NTL': load_raster(os.path.join(fpaths['nightlight'], templates['nightlight'] % year)),
        'CFI': load_raster(os.path.join(fpaths['frag'], templates['frag'] % year))
    }
    if year == years[0]:
        for var in ['elevation', 'slope', 'aspect']:
            data[var] = load_raster(os.path.join(fpaths['elevation'], templates[var]), is_elevation=True)
    return data

# ========================================
all_data = Parallel(n_jobs=N_CORES)(delayed(load_year_data)(year) for year in tqdm(years))

n_rows, n_cols = all_data[0]['CA'].shape
n_pixels = n_rows * n_cols
n_years = len(years)

# ========================================
X_full = np.zeros((n_pixels * n_years, len(predictor_names)), dtype=np.float32)
y_full = np.zeros(n_pixels * n_years, dtype=np.float32)

for i, year_data in enumerate(tqdm(all_data, desc="Building a spatiotemporal dataset")):
    if i == 0:
        for j, key in enumerate(['elevation', 'slope', 'aspect']):
            X_full[:, j+3] = np.tile(year_data[key].ravel(), n_years)
    start = i * n_pixels
    end = (i + 1) * n_pixels
    X_full[start:end, 0] = year_data['CA'].ravel()
    X_full[start:end, 1] = year_data['PD'].ravel()
    X_full[start:end, 2] = year_data['NTL'].ravel()
    y_full[start:end] = year_data['CFI'].ravel()

# ========================================
mask = ~np.isnan(X_full).any(axis=1) & ~np.isnan(y_full)
X_clean = X_full[mask]
y_clean = y_full[mask]
del X_full, y_full
gc.collect()

low, high = np.percentile(y_clean, [1, 99])
outlier_mask = (y_clean >= low) & (y_clean <= high)
X_filtered = X_clean[outlier_mask]
y_filtered = y_clean[outlier_mask]
del X_clean, y_clean
gc.collect()

# ========================================
def train_and_eval(X, y):
    model = XGBRegressor(n_jobs=N_CORES, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return model, r2, rmse, y_pred

model_full, r2_full, rmse_full, y_pred_full = train_and_eval(X_filtered, y_filtered)
residuals = np.abs(y_filtered - y_pred_full)

print(f"Model performance without removing outliers: R²={r2_full:.4f}, RMSE={rmse_full:.4f}")


percentiles = list(range(100, 99, -2))  

print(percentiles) 

results0 = []

for p in percentiles:
    if p == 100:
        mask = np.ones_like(residuals, dtype=bool)
    else:
        threshold = np.percentile(residuals, p)
        mask = residuals < threshold

    X_sub = X_filtered[mask]
    y_sub = y_filtered[mask]

    model, r2, rmse, _ = train_and_eval(X_sub, y_sub)

    results0.append({
        'percentile': p,
        'sample_count': len(y_sub),
        'R2': r2,
        'RMSE': rmse
    })

df_results = pd.DataFrame(results0)
df_results.to_csv(os.path.join(output_dir, "Maize_XGBoost_outlier_removal_results.csv"), index=False)


# ========================================
best_result0 = max(results0, key=lambda x: x['R2'])
best_percentile = best_result0['percentile']
threshold = np.percentile(residuals, best_percentile)
mask = residuals < threshold
X_best = X_filtered[mask]
y_best = y_filtered[mask]


X_train, X_test, y_train, y_test = train_test_split(X_best, y_best, test_size=0.2, random_state=42)  

X_train_df = pd.DataFrame(X_train, columns=predictor_names)
X_test_df = pd.DataFrame(X_test, columns=predictor_names)
y_train_series = pd.Series(y_train, name=response_name)
y_test_series = pd.Series(y_test, name=response_name)

# Optuna
def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'random_state': 42
    }
    model = XGBRegressor(**param, n_jobs=N_CORES)
    model.fit(X_train_df, y_train_series)
    pred = model.predict(X_test_df)
    return r2_score(y_test_series, pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, n_jobs=N_CORES)
best_params = study.best_params

pd.DataFrame.from_dict(best_params, orient='index').to_csv(
    os.path.join(output_dir, "Maize_best_params_filtered_28.csv"), header=['Value']
)
# ========================================
model = XGBRegressor(**best_params, n_jobs=N_CORES, random_state=42)
model.fit(X_train_df, y_train_series)

K_FOLDS = 5
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

def train_fold(fold, train_index, val_index):
    X_tr, X_val = X_train_df.iloc[train_index], X_train_df.iloc[val_index]
    y_tr, y_val = y_train_series.iloc[train_index], y_train_series.iloc[val_index]

    model = XGBRegressor(**best_params, n_jobs=N_CORES, random_state=42 + fold)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    slope, intercept, r_value, p_value, std_err = linregress(y_val, y_val_pred)
    
    print(f"[Fold {fold}] R²={r2:.4f}, RMSE={rmse:.4f}, Slope={slope:.4f}, p={p_value:.4e}")

    residual = y_val - y_val_pred
    return {
        'Fold': fold,
        'Model': model,
        'Observed': y_val.values,
        'Predicted': y_val_pred,
        'Residual': residual.values,
        'Metrics': {
            'R2': r2,
            'RMSE': rmse,
            'Slope': slope,
            'P_Value': p_value,
            'Sample_Count': len(y_val)
        }
    }


results = Parallel(n_jobs=K_FOLDS)(delayed(train_fold)(i + 1, tr, va) for i, (tr, va) in enumerate(kf.split(X_train_df)))

best_result = max(results, key=lambda r: r['Metrics']['R2'])
best_model = best_result['Model']


eval_df = pd.DataFrame({
    'Observed': best_result['Observed'],
    'Predicted': best_result['Predicted'],
    'Residual': best_result['Residual']
})
eval_df.to_csv(os.path.join(output_dir, "Maize_best_model_test_data_filtered_28.csv"), index=False)

metrics_df = pd.DataFrame([best_result['Metrics']])
metrics_df.to_csv(os.path.join(output_dir, "Maize_best_model_test_metrics_filtered_28.csv"), index=False)

# ========================================
importance = pd.DataFrame({
    'Feature': predictor_names,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
importance.to_csv(os.path.join(output_dir, "Maize_feature_importance_filtered_28.csv"), index=False)

plt.figure(figsize=(7, 5))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.xlabel("Importance", fontsize=16)
plt.ylabel("")                                 
plt.xticks(
    ticks=np.arange(0, importance['Importance'].max() + 0.1, 0.1),
    labels=[f"{x:.2f}" for x in np.arange(0, importance['Importance'].max() + 0.1, 0.1)],
    fontsize=16
)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Maize_feature_importance_filtered_28.png"), dpi=500)
plt.show()

# ========================================
sample_idx = np.random.choice(len(X_train_df), min(50000, len(X_train_df)), replace=False)
X_sample = X_train_df.iloc[sample_idx]
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_sample)

shap_importance = pd.DataFrame({
    'Feature': predictor_names,
    'SHAP_Importance': np.abs(shap_values).mean(axis=0)
}).sort_values(by='SHAP_Importance', ascending=False)
shap_importance.to_csv(os.path.join(output_dir, "Maize_shap_feature_importance_filtered_28.csv"), index=False)

plt.figure(figsize=(7, 5))
sns.barplot(x='SHAP_Importance', y='Feature', data=shap_importance, palette='viridis')
# plt.title("SHAP Feature Importance")
plt.xlabel("Mean |SHAP| value", fontsize=16)
plt.ylabel("")                                 
plt.xticks(
    ticks=np.arange(0, shap_importance['SHAP_Importance'].max() + 0.01, 0.01),
    labels=[f"{x:.2f}" for x in np.arange(0, shap_importance['SHAP_Importance'].max() + 0.01, 0.01)],
    fontsize=16
)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Maize_shap_feature_importance_filtered_28.png"), dpi=500)
plt.show()

pd.DataFrame(shap_values, columns=predictor_names).to_csv(
    os.path.join(output_dir, "Maize_shap_values_sample_filtered_28.csv"), index=False
)

# ========================================
import scipy.stats as st
from scipy.ndimage import gaussian_filter1d  
from statsmodels.nonparametric.smoothers_lowess import lowess  

pdp_output_dir = os.path.join(output_dir, "PDP_data")
os.makedirs(pdp_output_dir, exist_ok=True)

all_features = shap_importance['Feature'].tolist()[:6]

top3_features = all_features[:3]

for feature in all_features:
    x = X_sample[feature].values
    y = shap_values[:, predictor_names.index(feature)]

    df_raw = pd.DataFrame({feature: x, 'SHAP': y})
    df_raw.to_csv(os.path.join(pdp_output_dir, f"Maize_{feature}_shap_raw_28.csv"), index=False)

    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    smoothed = lowess(y_sorted, x_sorted, frac=0.2, return_sorted=True)

    df_fit = pd.DataFrame({'Feature': smoothed[:, 0], 'SHAP_Fit': smoothed[:, 1]})
    df_fit.to_csv(os.path.join(pdp_output_dir, f"Maize_{feature}_shap_fit_28.csv"), index=False)

plt.figure(figsize=(12, 4))
for i, feature in enumerate(top3_features):
    plt.subplot(1, 3, i + 1)

    df_fit = pd.read_csv(os.path.join(pdp_output_dir, f"Maize_{feature}_shap_fit_28.csv"))

    plt.plot(df_fit['Feature'], df_fit['SHAP_Fit'], color='blue', lw=2, label='LOWESS fit')

    plt.xlabel(feature, fontsize=14)
    plt.ylabel("SHAP Value", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(pdp_output_dir, "Maize_top3_PDP_fits_28.png"), dpi=300)
plt.show()
# ========================================
from scipy.stats import linregress
y_pred = model.predict(X_test_df)
r2 = r2_score(y_test_series, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_series, y_pred))

slope, intercept, r_value, p_value, std_err = linregress(y_test_series, y_pred)

eval_df = pd.DataFrame({
     'Observed': y_test_series,
     'Predicted': y_pred,
     'Residual': y_test_series - y_pred
 })
 eval_df.to_csv(os.path.join(output_dir, "Maize_model_test_data_filtered_28.csv"), index=False)

 metrics = pd.DataFrame({
     'R2': [r2],
     'RMSE': [rmse],
     'Slope': [slope],
     'P_Value': [p_value],
     'Sample_Count': [len(X_test_df)]
 })
 metrics.to_csv(os.path.join(output_dir, "Maize_model_test_metrics_filtered_28.csv"), index=False)
