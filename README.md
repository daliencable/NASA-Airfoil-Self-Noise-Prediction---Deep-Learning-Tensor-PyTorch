# Airfoil Self-Noise Prediction — Aerodynamic & Acoustic ML Project

Regression models trained on the NASA Airfoil Self-Noise dataset to predict **scaled sound pressure level (dB)** from aerodynamic test conditions. Implements and compares a baseline sklearn linear regression, a TensorFlow/Keras neural network, and a PyTorch neural network — each trained, evaluated, and iteratively improved.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Airfoil Self-Noise](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise)

The dataset contains 1,503 aeroacoustic test records from NASA wind tunnel experiments on NACA 0012 airfoil sections across a range of wind speeds and attack angles.

| Feature | Description |
|---|---|
| `frequency` | Frequency of sound (Hz) |
| `attack-angle` | Angle of attack (degrees) |
| `chord-length` | Chord length (meters) |
| `free-stream-velocity` | Free-stream flow velocity (m/s) |
| `suction-side-displacement thickness` | Boundary layer thickness (meters) |
| `scaled-sound-pressure` | **Target** — Scaled sound pressure level (dB) |

---

## Project Structure

```
├── Project1_Cable_Dalien.ipynb   # Main notebook
├── airfoil_self_noise.dat        # Raw dataset (tab-delimited, no header)
└── README.md
```

---

## Workflow

### 1. Data Loading & Exploration
- Load tab-delimited `.dat` file using `pandas`
- Assign column names
- Inspect dtypes, summary statistics, and null values
- Visualize feature relationships with a seaborn pairplot

### 2. Preprocessing
- Split features (`X`) and target (`y`)
- 80/20 train-test split with `random_state=1978`
- Apply `MinMaxScaler` to normalize feature values (fit on train, transform both sets)

### 3. Baseline — Sklearn Linear Regression (no scaling)
- Fit a basic `LinearRegression` model on unscaled training data
- Evaluate with R² score as a performance baseline

### 4. Model 1 — TensorFlow / Keras Neural Network
- Architecture: `Dense(32, relu)` → `Dense(1)`
- Optimizer: Adam | Loss: MSE | Batch size: 128 | Epochs: 150
- **Improved version:** `Dense(128, relu)` → `Dense(1)` (same hyperparameters)
- Evaluated with MSE, RMSE, and MAE

### 5. Model 2 — PyTorch Neural Network
- Architecture: `Linear(5→64)` → ReLU → `Linear(64→1)`
- Optimizer: Adam (lr=0.05) | Loss: MSELoss | Epochs: 150
- **Improved version:** hidden size increased to 128, retrained for 100 epochs
- Evaluated with MSE, RMSE, and MAE

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow torch
```

Python 3.12+ recommended (developed on conda base environment).

---

## Results Summary

| Model | Configuration | Key Metric |
|---|---|---|
| Linear Regression | Baseline, unscaled | R² score |
| TensorFlow NN | Dense(32) → Dense(1) | MSE / RMSE / MAE |
| TensorFlow NN (improved) | Dense(128) → Dense(1) | MSE ~578.49, RMSE ~24.05, MAE ~19.70 |
| PyTorch NN | hidden=64, 150 epochs | MSE / RMSE / MAE |
| PyTorch NN (improved) | hidden=128, 100 epochs | MSE / RMSE / MAE |

---

## Usage

1. Clone the repo and place `airfoil_self_noise.dat` in the same directory as the notebook
2. Launch Jupyter and open `Project1_Cable_Dalien.ipynb`
3. Run all cells top to bottom

---

## Author

Dalien Cable — M.S. AI/ML Program
