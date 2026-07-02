"""
Aerodynamic Drag Coefficient Surrogate — Gaussian Process Regression
=====================================================================
Replaces expensive CFD simulations with a GPR surrogate to predict
vehicle drag coefficient (Cd) from geometric design parameters.
Author: Bharath Kanaiah Parthiban
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os, json

np.random.seed(99)
sns.set_theme(style="whitegrid", font_scale=1.05)
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Synthetic CFD Dataset (DrivAer model-inspired) ──
N = 600
rng = np.random.default_rng(99)

front_area    = rng.uniform(2.0,  2.6,  N)   # m²
hood_angle    = rng.uniform(8.0,  18.0, N)   # deg
windshield_a  = rng.uniform(25,   38,   N)   # deg
roof_radius   = rng.uniform(0.3,  0.9,  N)   # m (rear roof curvature)
rear_angle    = rng.uniform(10,   30,   N)   # deg
underbody_h   = rng.uniform(0.10, 0.25, N)   # m (ground clearance)
wheel_exposed = rng.uniform(0.0,  1.0,  N)   # 0=enclosed, 1=open
mirror_width  = rng.uniform(0.12, 0.22, N)   # m

# Physics-informed Cd model
Cd = (
    0.30
    + 0.008 * (windshield_a - 30)
    + 0.004 * (rear_angle - 15)
    - 0.06 * roof_radius
    + 0.05 * wheel_exposed
    + 0.003 * (underbody_h - 0.15) * 100
    + 0.008 * mirror_width * 10
    + 0.0015 * hood_angle
    + rng.normal(0, 0.003, N)
)
Cd = np.clip(Cd, 0.18, 0.60)

features = ['Front Area (m²)', 'Hood Angle (°)', 'Windshield Angle (°)',
            'Roof Radius (m)', 'Rear Angle (°)', 'Ground Clearance (m)',
            'Wheel Exposure', 'Mirror Width (m)']
X = np.column_stack([front_area, hood_angle, windshield_a, roof_radius,
                     rear_angle, underbody_h, wheel_exposed, mirror_width])
y = Cd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

# GPR with Matérn kernel + noise
kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-3)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5, normalize_y=True)
gpr.fit(X_tr, y_train)

y_pred, y_std = gpr.predict(X_te, return_std=True)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
print(f"R²={r2:.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}")
json.dump({"R2": round(r2,4), "RMSE_Cd": round(rmse,5), "MAE_Cd": round(mae,5)},
          open(os.path.join(PLOT_DIR,'..','metrics.json'),'w'), indent=2)

# Fig 1 — Predicted vs Actual with uncertainty
sort_idx = np.argsort(y_test)
yt_s = y_test[sort_idx]; yp_s = y_pred[sort_idx]; ys_s = y_std[sort_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Aerodynamic Drag Surrogate — GPR Performance', fontsize=14, fontweight='bold')

ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.55, s=20, color='#7B1FA2', edgecolors='none')
lims = [y_test.min()-0.01, y_test.max()+0.01]
ax.plot(lims, lims, 'r--', lw=2, label='Ideal')
ax.set_xlabel('CFD Drag Coefficient (Cd)', fontsize=11)
ax.set_ylabel('GPR Predicted Cd', fontsize=11)
ax.set_title(f'Predicted vs Actual  |  R² = {r2:.4f}', fontsize=11)
ax.legend(); ax.set_xlim(lims); ax.set_ylim(lims)

ax = axes[1]
x_ax = np.arange(len(yt_s))
ax.plot(x_ax, yt_s, color='#1976D2', lw=1.5, label='CFD (True)')
ax.plot(x_ax, yp_s, color='#FF5722', lw=1.5, ls='--', label='GPR Mean')
ax.fill_between(x_ax, yp_s - 2*ys_s, yp_s + 2*ys_s, alpha=0.2, color='#FF5722', label='95% CI')
ax.set_xlabel('Test Sample (sorted by Cd)', fontsize=11)
ax.set_ylabel('Drag Coefficient (Cd)', fontsize=11)
ax.set_title('GPR Prediction with Uncertainty Bands', fontsize=11); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig1_gpr_performance.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig1")

# Fig 2 — Uncertainty distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(y_std, bins=35, color='#7B1FA2', alpha=0.8, edgecolor='white')
ax.axvline(y_std.mean(), color='red', ls='--', lw=2, label=f'Mean σ = {y_std.mean():.4f}')
ax.set_xlabel('Predictive Std Dev (σ)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('GPR Epistemic Uncertainty Distribution', fontsize=13, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig2_uncertainty.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig2")

# Fig 3 — Cd vs Windshield Angle (main driver)
angles = np.linspace(25, 38, 50)
X_line = np.zeros((50, 8))
X_line[:, 0] = 2.3       # fixed front area
X_line[:, 1] = 13        # fixed hood angle
X_line[:, 2] = angles    # sweep windshield
X_line[:, 3] = 0.6       # roof radius
X_line[:, 4] = 18        # rear angle
X_line[:, 5] = 0.17      # ground clearance
X_line[:, 6] = 0.3       # partial wheel exposure
X_line[:, 7] = 0.17      # mirror width
X_scaled = scaler.transform(X_line)
cd_pred, cd_std = gpr.predict(X_scaled, return_std=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(angles, cd_pred, color='#7B1FA2', lw=2.5, label='GPR Mean')
ax.fill_between(angles, cd_pred-2*cd_std, cd_pred+2*cd_std, alpha=0.2, color='#7B1FA2', label='95% CI')
ax.set_xlabel('Windshield Angle (°)', fontsize=11)
ax.set_ylabel('Predicted Drag Coefficient (Cd)', fontsize=11)
ax.set_title('Parametric Study: Windshield Angle vs Cd', fontsize=13, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig3_parametric_windshield.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig3")

# Fig 4 — Cd distribution of the design space
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(Cd, bins=40, color='#009688', alpha=0.8, edgecolor='white')
ax.axvline(Cd.mean(), color='red', ls='--', lw=2, label=f'Mean Cd = {Cd.mean():.3f}')
ax.axvline(0.28, color='blue', ls=':', lw=1.5, label='Sedan benchmark (0.28)')
ax.set_xlabel('Drag Coefficient (Cd)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Design Space Distribution of Cd', fontsize=13, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig4_cd_distribution.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig4 — Done!")
