# QBioCode

**A comprehensive suite of computational resources for quantum machine learning applications in healthcare and life sciences.**

[![PyPI version](https://badge.fury.io/py/qbiocode.svg)](https://badge.fury.io/py/qbiocode) [![Minimum Python Version](https://img.shields.io/badge/Python-%3E=%203.10-blue)](https://www.python.org/downloads/) [![Maximum Python Version Tested](https://img.shields.io/badge/Python-%3C=%203.12-blueviolet)](https://www.python.org/downloads/) [![Supported Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/) [![GitHub Pages](https://img.shields.io/badge/docs-sphinx-blue)](https://ibm.github.io/QBioCode/)

<img src="docs/source/img/QBioCode_logo.png" width="300" />

QBioCode provides tools for benchmarking quantum and classical machine learning models, analyzing data complexity, and making informed model selection decisions for healthcare and life science applications.

## 🌟 Key Features

- **QProfiler**: Automated ML benchmarking with data complexity analysis
- **QSage**: Meta-learning tool for intelligent model selection
- **Data Generation**: Create artificial datasets with controlled complexity
- **Quantum ML Support**: QSVC, PQK, VQC, QNN, Quantum Ensemble implementations
- **Classical ML Baselines**: RF, SVM, LR, DT, NB, MLP, XGBoost
- **Comprehensive Documentation**: Detailed tutorials and API reference

## 📋 Requirements

QBioCode requires Python **3.10 or higher** and has been tested with Python versions **3.10, 3.11, and 3.12**.

**Note:** Python 3.10+ is required for compatibility with the latest Qiskit ecosystem (qiskit-ibm-runtime 0.44.0+).

## 🚀 Quick Start

### Installation

#### Install from PyPI (Recommended)

```bash
# Standard installation: QBioCode's library, applications, quantum,
# machine-learning, visualization, and tutorial runtime dependencies
pip install qbiocode

# Backward-compatible alias; QProfiler and QSage are already included above
pip install 'qbiocode[apps]'

# Install everything needed for application use, documentation development,
# testing, linting, formatting, and type checking
pip install 'qbiocode[all]'
```

Use the standard installation when running QBioCode, QProfiler, QSage, its
notebooks, or its library functions. The `apps` extra remains available for
backward compatibility but currently adds no packages beyond the standard
installation. The `all` installation is intended for contributors who need
every optional development tool; it is not required for normal use and is not
needed in runtime images such as the Galaxy interactive tool.

More focused contributor installations are also available:

```bash
# Build the Sphinx documentation locally
pip install -e '.[docs]'

# Run tests, linters, formatters, and type checking
pip install -e '.[dev]'
```

Building notebook documentation also requires the Pandoc executable. Install
it with your operating system's package manager, for example
`conda install -c conda-forge pandoc` on macOS or Linux. This is needed only
when rebuilding the documentation, not when running QBioCode or its tutorials.

A fresh repository clone already includes the prebuilt HTML documentation.
Open `docs/_build/html/index.html` in a browser to view it without installing
the `docs` extra or Pandoc. The current published documentation is also
available from the Documentation link at the top of this README.

#### Install with Conda

QBioCode will be available on conda-forge and bioconda after the initial release review process.

```bash
# Once available on conda-forge (recommended)
conda install -c conda-forge qbiocode

# Or from bioconda (includes bioinformatics dependencies)
conda install -c bioconda -c conda-forge qbiocode

# Create a new environment with qbiocode
conda create -n qbiocode -c conda-forge qbiocode
conda activate qbiocode
```

**Current Status**: Conda packages are pending submission to conda-forge and bioconda. In the meantime, use pip within a conda environment:

```bash
conda create -n qbiocode python=3.10
conda activate qbiocode
pip install qbiocode
```

#### Install from Source

```bash
# Clone the repository
git clone https://github.com/IBM/QBioCode.git
cd QBioCode

# Create virtual environment
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate

# Install QBioCode in editable mode
pip install -e .

# Backward-compatible alias; applications are part of the standard install
pip install -e '.[apps]'

# Install every optional contributor dependency
pip install -e '.[all]'
```

**macOS Users:** XGBoost requires OpenMP. In a Conda or Miniforge
environment, install the cross-platform runtime from conda-forge:
```bash
conda install -c conda-forge llvm-openmp
```

Alternatively, Homebrew users can install it with:
```bash
brew install libomp
pip install --force-reinstall xgboost
```

For detailed installation instructions, see the [Installation Guide](https://ibm.github.io/QBioCode/installation.html).

### Running Tests

```bash
# Install the package with development dependencies
pip install -e '.[dev]'

# Run the test suite
python -m pytest
```

The current test suite focuses on utility modules and data-generation helpers that do not require a full runtime setup for all optional quantum workflows.

### Basic Usage

```python
import qbiocode as qbc

# Generate artificial data
qbc.generate_data(
    type_of_data='moons',
    save_path='data/moons',
    n_samples=[100, 200],
    noise=[0.1, 0.2],
    random_state=42
)

# Run QProfiler
from qbiocode.apps.qprofiler import qprofiler
import yaml

config = yaml.safe_load(open('configs/config.yaml'))
qprofiler.main(config)
```

## 📚 Applications

### QProfiler

**Automated ML Benchmarking with Data Complexity Analysis**

QProfiler provides a comprehensive benchmarking pipeline that:
- Evaluates both classical and quantum ML models
- Computes 15+ data complexity metrics
- Correlates model performance with data characteristics
- Generates detailed performance reports and visualizations

**Before you run — the input data must exist.** QProfiler reads a folder of CSV
datasets. The `folder_path` in the config is resolved **relative to the
`QBioCode` repo root** (the code truncates your current directory at `QBioCode`
and joins `folder_path` to it). So:

1. Run the command from **inside the QBioCode repo tree** (your working directory
   path must contain `QBioCode`).
2. The data must already be present at `<QBioCode-repo-root>/<folder_path>`.
   QProfiler does **not** create it. The tutorial dataset already ships in the
   repo at `tutorial/QProfiler/data/ld_data`.

**Usage:**
```bash
# Always run from inside the QBioCode repo so folder_path resolves correctly
cd /path/to/QBioCode

# 1) Run with the bundled default config
#    (expects data at <repo>/tutorial_test_data/lower_dim_datasets)
qprofiler

# 2) Point at the tutorial config + bundled data
#    (folder_path in this config = tutorial/QProfiler/data/ld_data)
qprofiler --config-dir=tutorial/QProfiler/configs --config-name=config

# 3) Override any value inline (Hydra syntax — key=value, no leading --)
#    folder_path is relative to the QBioCode repo root
qprofiler folder_path=tutorial/QProfiler/data/ld_data file_dataset=ALL backend=simulator
```

Results are written to `results/<config_file_name>/dataset=<file_dataset>/<backend>_<timestamp>/`
under the repo root. Run `qprofiler --help` to see every overridable key.

```python
# Python API
from qbiocode.apps.qprofiler import qprofiler
qprofiler.main(config)
```

[📖 QProfiler Documentation](https://ibm.github.io/QBioCode/apps/profiler.html) | [📓 Tutorial](tutorial/QProfiler/example_qprofiler.ipynb)

### QSage

**Intelligent Model Selection via Meta-Learning**

QSage uses surrogate models trained on extensive benchmarking data to:
- Predict model performance without running experiments
- Recommend best models based on dataset characteristics
- Save computational resources
- Provide interpretable predictions

**Usage:**
```bash
# Command line
qsage --data your_data.csv --output predictions.csv

# Python API
from qbiocode.apps.sage.sage import QuantumSage
sage = QuantumSage(data=benchmark_df, features=features, metrics=metrics)
predictions = sage.predict(new_dataset_features)
```

[📖 QSage Documentation](https://ibm.github.io/QBioCode/apps/sage.html) | [📓 Tutorial](tutorial/QSage/qsage.ipynb)

## 📖 Tutorials

Comprehensive Jupyter notebook tutorials are available:

### 1. [Artificial Data Generation](tutorial/Artificial_data_generation/example_data_generation.ipynb)
Learn how to create synthetic datasets with controlled properties:
- 2D manifolds (circles, moons, spirals)
- 3D manifolds (swiss_roll, s_curve, spheres)
- High-dimensional classification data
- Customizable complexity parameters

### 2. [QProfiler Tutorial](tutorial/QProfiler/example_qprofiler.ipynb)
Step-by-step guide to benchmarking ML models:
- Data generation and preparation
- Configuration setup
- Running QProfiler
- Analyzing results and visualizations
- Understanding data complexity metrics

### 3. [QSage Tutorial](tutorial/QSage/qsage.ipynb)
Learn to use meta-learning for model selection:
- Loading pre-trained QSage models
- Making predictions on new datasets
- Analyzing prediction accuracy
- Understanding feature importance

### 4. [Quantum Ensemble Learning](tutorial/QEnsemble/QEnsemble_example_blobs.ipynb)
Learn quantum ensemble methods for improved classification:
- Fixed swap-based ensemble approach
- Random unitary-based ensemble approach
- Quantum superposition for evaluating multiple training configurations
- Comparison with classical ensemble methods

### 5. [Quantum Projection Learning](tutorial/Quantum_Projection_Learning/QPL_example.ipynb)
Advanced quantum ML techniques with classical baselines.

## 🔧 Core Modules

### Data Generation
```python
import qbiocode as qbc

# Generate various dataset types
qbc.generate_data(type_of_data='circles', ...)
qbc.generate_data(type_of_data='moons', ...)
qbc.generate_data(type_of_data='classes', ...)
```

### Machine Learning Models

**Classical Models:**
- Random Forest (RF)
- Support Vector Machine (SVM)
- Logistic Regression (LR)
- Decision Tree (DT)
- Naive Bayes (NB)
- Multi-Layer Perceptron (MLP)
- XGBoost

**Quantum Models:**
- Quantum Support Vector Classifier (QSVC)
- Projected Quantum Kernel (PQK)
- Variational Quantum Classifier (VQC)
- Quantum Neural Network (QNN)
- Quantum Ensemble (QEnsemble) - swap and random unitary methods

**Quantum Ensemble Usage:**
```python
from qbiocode.learning import compute_qensemble
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Run quantum ensemble with swap method
results_swap = compute_qensemble(
    X_train, X_test, y_train, y_test,
    ensemble_method='swap',
    n_ensemble=4,
    seed=42
)

# Run quantum ensemble with random unitary method
results_random = compute_qensemble(
    X_train, X_test, y_train, y_test,
    ensemble_method='random_unitary',
    n_ensemble=4,
    seed=42
)
```

### Embeddings
- PCA, LLE, Isomap, Spectral Embedding
- UMAP, NMF
- Autoencoder

### Evaluation
- Model performance metrics (accuracy, F1, AUC)
- Data complexity analysis
- Correlation studies

## 🛠️ Utilities

### QML Config Generation

Generate configuration files for quantum model hyperparameter tuning:

```python
from qbiocode.utils import generate_qml_experiment_configs

num_configs, used_files = generate_qml_experiment_configs(
    template_config_path='configs/config.yaml',
    output_dir='configs/qml_gridsearch',
    data_dirs=['data/my_datasets'],
    qmethods=['qnn', 'vqc', 'qsvc'],
    reps=[1, 2],
    n_components=[5, 10],
    embeddings=['none', 'pca', 'isomap']
)
```

## 📊 Documentation

Full documentation is available at: **[https://ibm.github.io/QBioCode/](https://ibm.github.io/QBioCode/)**

- [Installation Guide](https://ibm.github.io/QBioCode/installation.html)
- [API Reference](https://ibm.github.io/QBioCode/api/qbiocode.html)
- [Tutorials](https://ibm.github.io/QBioCode/tutorials.html)
- [Background](https://ibm.github.io/QBioCode/background.html)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you use QBioCode in your research, please cite:

```bibtex
@software{qbiocode2024,
  title = {QBioCode: Quantum Machine Learning for Healthcare and Life Sciences},
  author = {Raubenolt, Bryan and Bose, Aritra and Rhrissorrakrai, Kahn and 
            Utro, Filippo and Mohan, Akhil and Blankenberg, Daniel and Parida, Laxmi},
  year = {2024},
  url = {https://github.com/IBM/QBioCode}
}
```

See [CITATION.cff](CITATION.cff) for more details.

## 👥 Authors

**Core Contributors:**

- Bryan Raubenolt (raubenb@ccf.org) - Cleveland Clinic
- Aritra Bose (a.bose@ibm.com) - IBM Research
- Kahn Rhrissorrakrai (krhriss@us.ibm.com) - IBM Research
- Filippo Utro (futro@us.ibm.com) - IBM Research
- Akhil Mohan (mohana2@ccf.org) - Cleveland Clinic
- Daniel Blankenberg (blanked2@ccf.org) - Cleveland Clinic
- Laxmi Parida (parida@us.ibm.com) - IBM Research

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on [GitHub](https://github.com/IBM/QBioCode/issues)
- Check the [documentation](https://ibm.github.io/QBioCode/)
- Contact the authors

---

**QBioCode** - Advancing quantum machine learning for healthcare and life sciences 🧬⚛️
