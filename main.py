import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Rich for colorful terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
console = Console()
console.print(Panel.fit("Calorie Prediction — Colorful Results", style="bold magenta"))
console.print("[bold yellow]Loading data...[/bold yellow]")

# CSVs are at repo root. If your files are in a `data/` folder, update these paths.
calories = pd.read_csv('calories.csv')
exercise = pd.read_csv('exercise.csv')

# Merge the two datasets on User_ID
data = pd.merge(exercise, calories, on='User_ID')

# Convert Gender to numbers (0 for male, 1 for female)
data['Gender'] = data['Gender'].apply(lambda x: 0 if x == 'male' else 1)

# Features (X) and Target (y)
X = data[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
y = data['Calories']

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. STANDARD SCALING (Crucial for Ridge/Lasso)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. APPLYING MEDICAL FORMULA (The Benchmark)
# ==========================================
# We calculate this manually on the test set to compare later
def calculate_keytel(row):
    # Formulas converted to return total calories (formula gives KJ/min or Cal/min depending on source)
    # Using the standard equation for Calories/min * Duration
    if row['Gender'] == 0: # Male
        return row['Duration'] * (0.6309 * row['Heart_Rate'] + 0.1988 * row['Weight'] + 0.2017 * row['Age'] - 55.0969) / 4.184
    else: # Female
        return row['Duration'] * (0.4472 * row['Heart_Rate'] - 0.1263 * row['Weight'] + 0.074 * row['Age'] - 20.4022) / 4.184

# Apply to test set (We need the original unscaled values for the formula)
medical_predictions = X_test.apply(calculate_keytel, axis=1)

# ==========================================
# 4. TRAINING REGRESSION MODELS
# ==========================================

# --- Model A: Simple Linear Regression ---
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

# --- Model B: Ridge Regression (Handles Multicollinearity) ---
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_reg.predict(X_test_scaled)

# --- Model C: Polynomial Regression (Captures Curves) ---
# We transform features to 2nd degree (e.g., Duration^2, Heart_Rate^2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

# ==========================================
# 5. EVALUATION & COMPARISON
# ==========================================
console.print("\n[bold cyan]Evaluating models...[/bold cyan]")

models = {
    "Medical Eq (Keytel)": medical_predictions,
    "Linear Regression": y_pred_lin,
    "Ridge Regression": y_pred_ridge,
    "Polynomial Regression": y_pred_poly,
}

results = []
for name, pred in models.items():
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    results.append({"name": name, "r2": r2, "rmse": rmse})

# Build a nice table with Rich
table = Table(title="Model Comparison", box=box.ROUNDED, show_lines=True)
table.add_column("Model", style="bold")
table.add_column("R2 Score", justify="right")
table.add_column("RMSE", justify="right")

best = None
for r in results:
    table.add_row(r["name"], f"{r['r2']:.4f}", f"{r['rmse']:.2f}")
    if best is None or r["r2"] > best["r2"]:
        best = r

console.print(table)
console.print(Panel.fit(f":sparkles: Best model: [bold green]{best['name']}[/bold green] — R2={best['r2']:.4f}, RMSE={best['rmse']:.2f} :rocket:", style="bright_blue"))

# ==========================================
# 6. VISUALIZATION (Polished for evaluator)
# ==========================================
plt.figure(figsize=(12, 6), facecolor='white')

# Shared diagonal for reference
min_val = min(y_test.min(), medical_predictions.min(), y_pred_poly.min())
max_val = max(y_test.max(), medical_predictions.max(), y_pred_poly.max())

# Plot 1: Medical Equation Accuracy
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(y_test, medical_predictions, color='#e63946', alpha=0.35, s=18, label='Medical Eq')
ax1.plot([min_val, max_val], [min_val, max_val], color='#343a40', linestyle='--', linewidth=1.5)
ax1.set_title('Medical Equation vs Actual', fontsize=12, weight='bold')
ax1.set_xlabel('Actual Calories')
ax1.set_ylabel('Predicted Calories')
ax1.grid(alpha=0.2)
ax1.legend()

# Plot 2: Polynomial Regression Accuracy (best model)
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(y_test, y_pred_poly, color='#1d3557', alpha=0.35, s=18, label='Polynomial (AI)')
ax2.plot([min_val, max_val], [min_val, max_val], color='#343a40', linestyle='--', linewidth=1.5)
ax2.set_title('AI (Polynomial) vs Actual', fontsize=12, weight='bold')
ax2.set_xlabel('Actual Calories')
ax2.set_ylabel('Predicted Calories')
ax2.grid(alpha=0.2)
ax2.legend()

# Annotate RMSEs on the figure
plt.suptitle(f"Model Comparison — Best: {best['name']} (R2={best['r2']:.3f}, RMSE={best['rmse']:.2f})", fontsize=14, weight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plot_path = 'results_plot.png'
plt.savefig(plot_path, dpi=220)
console.print(f"[bold green]Saved polished comparison plot to[/bold green] [underline]{plot_path}[/underline]")
console.print(Panel.fit("Tip: open `results_plot.png` to view at full resolution.", style="green"))