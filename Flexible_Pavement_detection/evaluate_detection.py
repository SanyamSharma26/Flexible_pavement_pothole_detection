import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load ground truth and predictions
truth = pd.read_csv('ground_truth.csv')
pred = pd.read_csv('predictions.csv')

# Merge on frame
merged = pd.merge(truth, pred, on='frame', suffixes=('_true', '_pred'))

# Calculate regression metrics for area, width, height
mse_list, mae_list, r2_list = [], [], []
for col in ['area_cm2', 'width_cm', 'height_cm']:
    y_true = merged[f'{col}_true']
    y_pred = merged[f'{col}_pred']
    mse_list.append(mean_squared_error(y_true, y_pred))
    mae_list.append(mean_absolute_error(y_true, y_pred))
    r2_list.append(r2_score(y_true, y_pred))

system_mse = sum(mse_list) / len(mse_list)
system_mae = sum(mae_list) / len(mae_list)
system_r2 = sum(r2_list) / len(r2_list)

# Calculate detection accuracy (all metrics within 10% of ground truth)
correct = 0
for _, row in merged.iterrows():
    area_ok = abs(row['area_cm2_true'] - row['area_cm2_pred']) / max(row['area_cm2_true'], 1e-6) < 0.1
    width_ok = abs(row['width_cm_true'] - row['width_cm_pred']) / max(row['width_cm_true'], 1e-6) < 0.1
    height_ok = abs(row['height_cm_true'] - row['height_cm_pred']) / max(row['height_cm_true'], 1e-6) < 0.1
    if area_ok and width_ok and height_ok:
        correct += 1
accuracy = correct / len(merged) if len(merged) > 0 else 0

print('--- System Evaluation Metrics ---')
print(f'Accuracy: {accuracy:.2%}')
print(f'MSE: {system_mse:.4f}')
print(f'MAE: {system_mae:.4f}')
print(f'R²: {system_r2:.4f}') 