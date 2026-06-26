import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


# 1. LOAD & PREPROCESS DATASET

data = pd.read_csv("dmart_sales.csv")
data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
data['Month']      = data['Order Date'].dt.month
data['DayOfWeek']  = data['Order Date'].dt.dayofweek

def get_season(month):
    if month in [3, 4, 5, 6]:   return "Summer"
    elif month in [7, 8, 9]:    return "Monsoon"
    else:                        return "Winter"

data['Season'] = data['Month'].apply(get_season)
data = data.drop(columns=['Order ID', 'Customer Name', 'Order Date', 'State'], errors='ignore')
data = data.dropna()


# 2. FEATURES & TARGET

X = data.drop(columns=['Sales'])
y = data['Sales']

cat_cols = ['Category', 'Sub Category', 'City', 'Region', 'Season']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. MODEL 1: LINEAR REGRESSION

lr_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())])
lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

lr_mae  = round(mean_absolute_error(y_test, lr_pred), 2)
lr_rmse = round(np.sqrt(mean_squared_error(y_test, lr_pred)), 2)
lr_r2   = round(r2_score(y_test, lr_pred), 4)

print("=" * 50)
print("  LINEAR REGRESSION")
print("=" * 50)
print(f"  MAE  : {lr_mae}")
print(f"  RMSE : {lr_rmse}")
print(f"  R²   : {lr_r2}  ({lr_r2*100:.1f}% variance explained)")


# 4. MODEL 2: DECISION TREE

dt_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor(random_state=42))])
dt_pipeline.fit(X_train, y_train)
dt_pred = dt_pipeline.predict(X_test)

dt_mae  = round(mean_absolute_error(y_test, dt_pred), 2)
dt_rmse = round(np.sqrt(mean_squared_error(y_test, dt_pred)), 2)
dt_r2   = round(r2_score(y_test, dt_pred), 4)

print("\n" + "=" * 50)
print("  DECISION TREE")
print("=" * 50)
print(f"  MAE  : {dt_mae}")
print(f"  RMSE : {dt_rmse}")
print(f"  R²   : {dt_r2}  ({dt_r2*100:.1f}% variance explained)")


# 5. BUILD EVALUATION REPORT DF

eval_df = X_test.copy().reset_index(drop=True)
eval_df['Actual']       = y_test.values
eval_df['LR_Predicted'] = lr_pred
eval_df['DT_Predicted'] = dt_pred
eval_df['LR_Residual']  = y_test.values - lr_pred   # positive = underpredicted
eval_df['DT_Residual']  = y_test.values - dt_pred
eval_df['LR_AbsError']  = np.abs(eval_df['LR_Residual'])
eval_df['DT_AbsError']  = np.abs(eval_df['DT_Residual'])

# Error buckets for LR
bins   = [0, 200, 500, 1000, 2000, float('inf')]
labels = ['<200', '200-500', '500-1K', '1K-2K', '>2K']
eval_df['LR_ErrorBucket'] = pd.cut(eval_df['LR_AbsError'], bins=bins, labels=labels)


# 6. SUMMARY STATS

print("\n" + "=" * 50)
print("  MODEL COMPARISON SUMMARY")
print("=" * 50)
print(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print(f"  {'-'*46}")
print(f"  {'Linear Regression':<22} {lr_mae:>8} {lr_rmse:>8} {lr_r2:>8}")
print(f"  {'Decision Tree':<22} {dt_mae:>8} {dt_rmse:>8} {dt_r2:>8}")
print(f"\n  ✅ Better model: {'Linear Regression' if lr_mae < dt_mae else 'Decision Tree'}")

# Error analysis by Category
print("\n  📊 LR Mean Absolute Error by Category:")
cat_err = eval_df.groupby('Category')['LR_AbsError'].mean().sort_values(ascending=False)
for cat, err in cat_err.items():
    bar = "█" * int(err / 50)
    print(f"    {cat:<20} ₹{err:>8.0f}  {bar}")

# Error bucket distribution
print("\n  📊 LR Error Distribution:")
bucket_pct = eval_df['LR_ErrorBucket'].value_counts(normalize=True).sort_index() * 100
for bucket, pct in bucket_pct.items():
    bar = "█" * int(pct / 5)
    print(f"    Error {bucket:<10} {pct:>5.1f}%  {bar}")


# 7. SAVE EVERYTHING

joblib.dump(lr_pipeline, "dmart_model.pkl")
print("\n✅ Linear Regression model saved → dmart_model.pkl")

joblib.dump(dt_pipeline, "dt_model.pkl")
print("✅ Decision Tree model saved     → dt_model.pkl")

# Save full metrics dict
metrics = {
    "lr": {"MAE": lr_mae, "RMSE": lr_rmse, "R2": lr_r2},
    "dt": {"MAE": dt_mae, "RMSE": dt_rmse, "R2": dt_r2}
}
joblib.dump(metrics, "model_metrics.pkl")
print("✅ Metrics saved                 → model_metrics.pkl")

# Save evaluation dataframe for dashboard report
joblib.dump(eval_df, "eval_report.pkl")
print("✅ Evaluation report saved       → eval_report.pkl")

# Keep old dt_metrics.pkl for backward compat
joblib.dump({"MAE": dt_mae, "RMSE": dt_rmse}, "dt_metrics.pkl")
print("✅ DT metrics saved              → dt_metrics.pkl")