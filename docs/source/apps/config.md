# QProfiler Configuration Guide

This guide explains how to configure QProfiler using YAML configuration files for reproducible experiments and batch processing.

## Overview

QProfiler uses YAML configuration files to define:
- Input datasets and output directories
- Machine learning models to evaluate
- Quantum backend settings
- Embedding methods and parameters
- Train/test split configuration
- Model hyperparameters

An example configuration file can be found at [`qbiocode/apps/qprofiler/configs/config.yaml`](https://github.com/qiskit-community/QBioCode/blob/main/qbiocode/apps/qprofiler/configs/config.yaml).

## Quick Start

Here's a minimal configuration to get started:

```yaml
# Basic configuration
config_file_name: 'my_experiment'
folder_path: 'data/'
file_dataset: 'my_dataset.csv'
seed: 42

# Models to evaluate
model: ['rf', 'svc', 'qsvc']

# Embedding
embeddings: ['none']
n_components: 3

# Train/test split
test_size: 0.2
stratify: ['y']
scaling: ['True']

# Quantum backend (for QML models)
backend: 'simulator'
shots: 1024
```

---

## Configuration Sections

### Input Data

Specify the location and selection of input datasets.

**Single Dataset:**

```yaml
config_file_name: 'experiment_name'
folder_path: 'data/'
file_dataset: 'dataset.csv'
```

**All Datasets in Folder:**

```yaml
folder_path: 'data/'
file_dataset: 'ALL'  # Process all CSV files in folder
```

**Multiple Specific Datasets:**

```yaml
file_dataset: ['dataset1.csv', 'dataset2.csv', 'dataset3.csv']
```

**Output Directory:**

```yaml
output_dir: 'results/'  # Where to save results
```

### Random Seeds

Set random seeds for reproducibility:

```yaml
seed: 42      # Seed for classical ML algorithms
q_seed: 42    # Seed for quantum algorithms
```

```{tip}
Always set seeds for reproducible experiments. Use the same seed across runs to compare results.
```

### Quantum Backend Configuration

Configure quantum computing backend for QML models (QSVC, VQC, QNN, PQK).

**Simulator (Default):**

```yaml
backend: 'simulator'
shots: 1024
```

Uses the Qiskit statevector simulator for exact, noiseless quantum simulation.

**AerSimulator with Custom Simulation Method:**

```yaml
backend: 'simulator_aer'
sim_method: 'statevector'  # Options: statevector, matrix_product_state, tensor_network, etc.
shots: 1024
```

Provides access to AerSimulator's various simulation methods. Useful for:
- **GPU acceleration**: Use `sim_method: 'tensor_network'` with GPU support
- **Memory efficiency**: Use `sim_method: 'matrix_product_state'` for low-entanglement circuits
- **Clifford circuits**: Use `sim_method: 'stabilizer'` for fast simulation

**Noisy Simulation Based on IBM Device:**

```yaml
backend: 'noisy_ibm_cleveland'  # Noisy simulation modeled on IBM device
sim_method: 'matrix_product_state'  # Simulation method for AerSimulator
shots: 1024
```

Simulates quantum circuits with realistic noise models from actual IBM Quantum devices. This feature:
- Extracts the noise model from a specified IBM device (e.g., `ibm_cleveland`, `ibm_kyoto`)
- Runs simulation locally using AerSimulator with the device's noise characteristics
- Allows testing quantum algorithms under realistic noise conditions without queue time
- Supports any simulation method available in AerSimulator

**Format:** `'noisy_<device_name>'` where `<device_name>` is any IBM Quantum device name.

**Examples:**
- `'noisy_ibm_cleveland'` - Noise model from IBM Cleveland
- `'noisy_ibm_kyoto'` - Noise model from IBM Kyoto
- `'noisy_ibm_sherbrooke'` - Noise model from IBM Sherbrooke

```{tip}
**Choosing a Simulation Method for Noisy Simulations:**

- **`matrix_product_state`**: Recommended for most quantum machine learning circuits. Efficient for circuits with moderate entanglement.
- **`tensor_network`**: Best for GPU-accelerated simulations. Requires `qiskit-aer-gpu` installation.
- **`statevector`**: Most accurate but memory-intensive. Limited to ~20-25 qubits depending on available RAM.
- **`automatic`**: Let AerSimulator choose the best method based on circuit properties.
```

**IBM Quantum Hardware:**

```yaml
backend: 'ibm_least'  # Use least busy device
# OR
backend: 'ibm_kyoto'  # Specific device name
shots: 4096
resil_level: 1  # Error mitigation level (1-3)
```

Runs circuits on actual IBM Quantum hardware.

**IBM Quantum Credentials:**

```yaml
qiskit_json_path: '~/.qiskit/qiskit-ibm.json'
name: 'account_qbc'  # Account alias in JSON
ibm_instance: 'ibm-q/open/main'  # Optional: specific instance
```

```{important}
**IBM Credentials Required for Noisy Simulations:**

When using `noisy_<device_name>` backends, you must provide valid IBM Quantum credentials even though the simulation runs locally. This is because QBioCode needs to:
1. Connect to IBM Quantum services to retrieve the device's noise model
2. Download the latest calibration data for accurate noise simulation

The actual circuit execution happens locally on your machine using AerSimulator, so you won't incur queue wait times or consume IBM Quantum compute credits.
```

```{note}
**Backend Options:**
- `'simulator'`: Local Qiskit statevector simulator (exact, noiseless)
- `'simulator_aer'`: AerSimulator with configurable simulation method
- `'noisy_<device_name>'`: Noisy simulation based on IBM device noise model (e.g., `'noisy_ibm_cleveland'`)
- `'ibm_least'`: Automatically select least busy IBM Quantum device
- `'ibm_<device_name>'`: Specific IBM Quantum device (e.g., 'ibm_kyoto')

**Simulation Methods (for AerSimulator):**
When using `'simulator_aer'` or `'noisy_<device_name>'` backends, specify the simulation method via `sim_method`:
- `'statevector'`: Exact statevector simulation (default, memory intensive)
- `'matrix_product_state'`: Efficient for low-entanglement circuits
- `'tensor_network'`: GPU-accelerated tensor network simulation
- `'stabilizer'`: Fast simulation for Clifford circuits
- `'extended_stabilizer'`: Approximate simulation for near-Clifford circuits
- `'automatic'`: Automatically select best method

**Shots:** Number of circuit executions. Higher = more accurate but slower.

**Resilience Level:** Error mitigation strength (1=light, 2=medium, 3=heavy). Higher = more accurate but slower.
```

### Embedding Methods

Dimensionality reduction techniques to apply before model training.

**No Embedding:**

```yaml
embeddings: ['none']
```

**Single Embedding Method:**

```yaml
embeddings: ['pca']
n_components: 3  # Reduce to 3 dimensions
```

**Multiple Embedding Methods:**

```yaml
embeddings: ['pca', 'nmf', 'umap', 'autoencoder']
n_components: 5
```

**Available Embedding Methods:**
- `'none'`: No dimensionality reduction
- `'pca'`: Principal Component Analysis
- `'nmf'`: Non-negative Matrix Factorization
- `'umap'`: Uniform Manifold Approximation and Projection
- `'autoencoder'`: Neural network autoencoder

```{tip}
Start with `'none'` to establish baseline performance, then try `'pca'` for faster quantum model training.
```

### Train/Test Split

Configure data splitting and preprocessing.

```yaml
test_size: 0.2      # 80% train, 20% test
stratify: ['y']     # Maintain class distribution
scaling: ['True']   # Standardize features
```

**Parameters:**
- `test_size`: Proportion of data for testing (0.0-1.0)
- `stratify`: `['y']` to maintain class balance, `['n']` for random split
- `scaling`: `['True']` to standardize features (recommended), `['False']` for raw data

```{warning}
Always use `stratify: ['y']` for imbalanced datasets to ensure both train and test sets have representative class distributions.
```

### Model Selection

Specify which machine learning models to evaluate.

**All Models:**

```yaml
model: ['svc', 'dt', 'lr', 'nb', 'rf', 'mlp', 'xgb', 'qsvc', 'vqc', 'qnn', 'pqk']
```

**Classical Models Only:**

```yaml
model: ['rf', 'svc', 'lr', 'mlp', 'xgb']
```

**Quantum Models Only:**

```yaml
model: ['qsvc', 'vqc', 'qnn', 'pqk']
```

**Available Models:**

| Model | Type | Description |
|-------|------|-------------|
| `svc` | Classical | Support Vector Classifier |
| `dt` | Classical | Decision Tree |
| `lr` | Classical | Logistic Regression |
| `nb` | Classical | Naive Bayes |
| `rf` | Classical | Random Forest |
| `mlp` | Classical | Multi-Layer Perceptron |
| `xgb` | Classical | XGBoost |
| `qsvc` | Quantum | Quantum Support Vector Classifier |
| `vqc` | Quantum | Variational Quantum Classifier |
| `qnn` | Quantum | Quantum Neural Network |
| `pqk` | Quantum | Projected Quantum Kernel |

### Model Hyperparameters

Configure hyperparameters for each model. Each model has:
- **Standard arguments**: Single values for quick runs
- **Grid search arguments**: Lists of values for hyperparameter tuning

**Example: Support Vector Classifier (SVC)**

```yaml
# Standard run with fixed parameters
svc_args:
  C: 1.0
  gamma: 0.1
  kernel: 'rbf'

# Grid search over parameter combinations
gridsearch_svc_args:
  C: [0.1, 1, 10, 100]
  gamma: [0.001, 0.01, 0.1, 1]
  kernel: ['linear', 'rbf', 'poly', 'sigmoid']
```

**Example: Random Forest (RF)**

```yaml
rf_args:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 2

gridsearch_rf_args:
  n_estimators: [50, 100, 200]
  max_depth: [5, 10, 15, 20]
  min_samples_split: [2, 5, 10]
```

**Example: XGBoost (XGB)**

```yaml
xgb_args:
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 6

gridsearch_xgb_args:
  n_estimators: [50, 100, 200]
  learning_rate: [0.01, 0.1, 0.3]
  max_depth: [3, 6, 9]
```

```{seealso}
For detailed parameter descriptions, see the scikit-learn documentation:
- [SVC Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Random Forest Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
```

### Quantum Model Hyperparameters

For quantum models, hyperparameter tuning requires generating separate config files for each combination.

```{important}
**QML Grid Search:**

Quantum model grid search is handled differently than classical models. Use the `generate_experiments.ipynb` notebook in `archive/tutorial_notebooks/qml_experiment_generators/` to generate individual config files for each parameter combination.

This approach is necessary because:
1. Quantum jobs are submitted to IBM Quantum queue
2. Each configuration may take hours to complete
3. Separate configs allow parallel job submission
```

**Example: Quantum SVC (QSVC)**

```yaml
qsvc_args:
  feature_map: 'ZZFeatureMap'
  reps: 2
  entanglement: 'linear'
```

**Example: Variational Quantum Classifier (VQC)**

```yaml
vqc_args:
  feature_map: 'ZZFeatureMap'
  ansatz: 'RealAmplitudes'
  reps: 3
  optimizer: 'COBYLA'
```

---

## Complete Example Configuration

Here's a comprehensive example combining all sections:

```yaml
# Experiment identification
config_file_name: 'comprehensive_experiment'

# Input data
folder_path: 'datasets/'
file_dataset: ['cancer_data.csv', 'diabetes_data.csv']
output_dir: 'results/experiment_001/'

# Reproducibility
seed: 42
q_seed: 42

# Quantum backend - Noisy simulation example
backend: 'noisy_ibm_cleveland'
sim_method: 'matrix_product_state'
shots: 1024
resil_level: 1
qiskit_json_path: '~/.qiskit/qiskit-ibm.json'
name: 'my_ibm_account'

# Dimensionality reduction
embeddings: ['none', 'pca']
n_components: 5

# Data splitting
test_size: 0.2
stratify: ['y']
scaling: ['True']

# Models to evaluate
model: ['rf', 'svc', 'mlp', 'xgb', 'qsvc', 'pqk']

# Classical model parameters
rf_args:
  n_estimators: 100
  max_depth: 10

gridsearch_rf_args:
  n_estimators: [50, 100, 200]
  max_depth: [5, 10, 15]

svc_args:
  C: 1.0
  kernel: 'rbf'

gridsearch_svc_args:
  C: [0.1, 1, 10]
  kernel: ['linear', 'rbf']

# Quantum model parameters
qsvc_args:
  feature_map: 'ZZFeatureMap'
  reps: 2
```

**Alternative Backend Configurations:**

```yaml
# For exact noiseless simulation
backend: 'simulator'

# For AerSimulator with GPU acceleration
backend: 'simulator_aer'
sim_method: 'tensor_network'

# For actual IBM Quantum hardware
backend: 'ibm_kyoto'
shots: 4096
resil_level: 2
```

---

## Best Practices

```{tip}
**Configuration Tips:**

1. **Start Simple**: Begin with a minimal config and add complexity gradually
2. **Use Descriptive Names**: Name configs by experiment purpose (e.g., `cancer_baseline.yaml`)
3. **Version Control**: Keep configs in git to track experiment history
4. **Document Changes**: Add comments in YAML to explain non-obvious choices
5. **Test Locally First**: Use `backend: 'simulator'` before submitting to quantum hardware
```

```{warning}
**Common Pitfalls:**

- **Missing Seeds**: Always set `seed` and `q_seed` for reproducibility
- **Too Many Grid Search Combinations**: Start with small grids to estimate runtime
```

---

## Troubleshooting

**Problem: "Config file not found"**
- Ensure config file is in `configs/` directory
- Check file name matches `--config-name` argument
- Use relative path from project root

**Problem: "Invalid backend"**
- Verify IBM Quantum credentials are configured
- Check device name spelling (use `ibm_<device>` format)
- Ensure you have access to the specified instance

**Problem: "Grid search taking too long"**
- Reduce number of parameter combinations
- Use fewer cross-validation folds
- Consider using `RandomizedSearchCV` for large grids

**Problem: "Out of memory"**
- Reduce `n_components` for embeddings
- Use smaller `test_size` to reduce data size
- Process datasets one at a time instead of batch

---

## See Also

- :doc:`QProfiler Usage Guide <profiler>` - How to run QProfiler
- :doc:`QSage Configuration <sage>` - Meta-learning model selection
- :doc:`Tutorial Notebooks <../tutorials>` - Step-by-step examples
