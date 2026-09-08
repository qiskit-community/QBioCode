# Contributing to QBioCode

Thank you for your interest in contributing to QBioCode! We welcome contributions from the community and are grateful for your support in making quantum more accessible for biological and healthcare applications.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Reporting Issues](#reporting-issues)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Coding Standards](#coding-standards)
- [Documentation](#documentation)
- [Testing](#testing)
- [Community](#community)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of quantum computing concepts (see our [Background](docs/source/background.md) documentation)

### Useful Resources

- **Documentation**: [QBioCode Docs](docs/source/index.rst)
- **Tutorials**: [Tutorial Notebooks](tutorial/)
- **API Reference**: [API Documentation](docs/source/api_overview.rst)
- **Qiskit Resources**: [Qiskit YouTube Channel](https://www.youtube.com/@qiskit)

## Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/QBioCode.git
   cd QBioCode
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv qbiocode-dev
   source qbiocode-dev/bin/activate  # On Windows: qbiocode-dev\Scripts\activate
   ```

3. **Install in Development Mode**
   ```bash
   pip install -e .
   ```
   `pyproject.toml` reads the runtime dependencies from
   `requirements/requirements-base.txt`, so this one command installs them too --
   there is no separate `pip install -r requirements.txt` step.

4. **Install Optional Tiers** (as needed)
   ```bash
   pip install -e ".[dev]"     # pytest, nbclient, black, mypy, build
   pip install -e ".[quvine]"  # the QuVINE embedding methods
   pip install -e ".[docs]"    # the Sphinx toolchain (pandoc is a system binary)
   pip install -e ".[all]"     # every tier at once
   ```
   Or install the complete development environment in one step, from the repo root:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify Installation**
   ```bash
   python -c "import qbiocode; print(qbiocode.__version__)"
   ```

6. **Building Distribution Artifacts Locally**
   ```bash
   rm -rf dist build *.egg-info && python -m build
   ```
   The `rm -rf` is not housekeeping. setuptools *unions* the previous
   `qbiocode.egg-info/SOURCES.txt` into each new sdist rather than recomputing it,
   so a checkout that ever built with a broader `MANIFEST.in` keeps shipping files
   `MANIFEST.in` no longer names -- and because `MANIFEST.in` globs the working
   tree, notebook run output under `tutorial/**/data/` lands in the archive too.
   Both make the artifact a function of your shell history rather than of the
   commit. `.github/workflows/release.yml` avoids this by construction, since a
   fresh `actions/checkout` has no stale metadata;
   `tests/integration/test_distribution_contents.py` guards it for everyone else.

## How to Contribute

We welcome various types of contributions:

### 🐛 Bug Fixes
- Fix existing bugs or issues
- Improve error handling
- Enhance code robustness

### ✨ New Features
- Implement new quantum algorithms
- Add classical ML baselines
- Develop new data generation methods
- Create visualization tools

### 📚 Documentation
- Improve existing documentation
- Add code examples
- Create tutorials or notebooks
- Fix typos or clarify explanations

### 🧪 Testing
- Write unit tests
- Add integration tests
- Improve test coverage
- Test on different platforms

### 🎨 Code Quality
- Refactor code for clarity
- Optimize performance
- Improve code organization
- Add type hints

## Reporting Issues

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** for answers
3. **Try the latest version** to see if the issue persists

### Creating a Good Issue

When reporting a bug, please include:

- **Clear title**: Descriptive and specific
- **Environment details**: OS, Python version, QBioCode version
- **Steps to reproduce**: Minimal code example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full traceback if applicable
- **Screenshots**: If relevant

**Example:**
```markdown
## Bug: QSVC fails with 3-qubit feature map

**Environment:**
- OS: macOS 14.0
- Python: 3.10.12
- QBioCode: 0.1.0
- Qiskit: 1.0.0

**Steps to reproduce:**
```python
from qbiocode.learning import compute_qsvc
# ... minimal code to reproduce
```

**Expected:** Model should train successfully
**Actual:** Raises ValueError: "Invalid feature map dimension"
**Traceback:** [paste full error]
```

## Submitting Pull Requests

### Before You Start

1. **Open an issue** to discuss major changes
2. **Check existing PRs** to avoid duplicate work
3. **Create a feature branch** from `main`

### Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

2. **Make Your Changes**
   - Write clear, documented code
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run existing tests
   python -m pytest
   
   # Check code style
   black qbiocode/
   flake8 qbiocode/
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add quantum feature map X"
   ```
   
   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `style:` Code style changes
   - `chore:` Maintenance tasks

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   
   Then create a pull request on GitHub with:
   - **Clear title** describing the change
   - **Description** explaining what and why
   - **Link to related issue** (e.g., "Closes #123")
   - **Testing details** showing it works
   - **Screenshots** if UI/visualization changes

### PR Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, your PR will be merged
- Your contribution will be acknowledged in release notes

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 88 characters (Black default)
- Use meaningful variable and function names

### Code Organization

```python
# Standard library imports
import os
from typing import List, Dict, Optional

# Third-party imports
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

# Local imports
from qbiocode.utils import helper_fn
```

### Documentation Style

Use Google-style docstrings:

```python
def compute_quantum_kernel(X: np.ndarray, feature_map: str = 'ZZFeatureMap') -> np.ndarray:
    """Compute quantum kernel matrix for input data.
    
    Args:
        X: Input data array of shape (n_samples, n_features)
        feature_map: Type of quantum feature map to use
        
    Returns:
        Kernel matrix of shape (n_samples, n_samples)
        
    Raises:
        ValueError: If X has invalid shape or feature_map is unknown
        
    Example:
        >>> X = np.random.rand(10, 4)
        >>> K = compute_quantum_kernel(X, feature_map='ZZFeatureMap')
        >>> K.shape
        (10, 10)
    """
    # Implementation
    pass
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Dict, Optional, Union
import numpy as np

def process_data(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """Process input data with optional normalization."""
    pass
```

## Documentation

### Building Documentation Locally

```bash
cd docs
make clean
make html
```

View documentation at `docs/build/html/index.html`

### Documentation Guidelines

- Update relevant `.rst` or `.md` files
- Add docstrings to all public functions/classes
- Include code examples in docstrings
- Update tutorials if adding new features
- Add references to scientific papers when applicable

### Adding Tutorials

1. Create the Jupyter notebook in the `tutorial/` directory -- the single source.
   Do **not** copy it into `docs/source/tutorials/`: that tree is generated from
   `tutorial/` by `_sync_tutorials()` in `docs/source/conf.py` on every build, and
   is gitignored. Hand-maintaining a second copy is exactly how the two trees
   drifted before.
2. Add it to the hidden toctree at the bottom of `docs/source/tutorials.md`, as
   `tutorials/<subdir>/<name>`, and write its gallery entry in the same file.
   A notebook that no toctree names becomes an orphan page, which fails the
   `-W` docs build; `tests/test_docs_structure.py` catches it first.
3. Read fixtures through `qbiocode.utils.tutorial_data_path("<file>")` rather than
   a hand-built relative path, so the notebook runs from any working directory.
4. Test notebook execution, and commit it with its outputs -- `nbsphinx_execute`
   is `'never'`, so the committed outputs are what the site publishes.

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=qbiocode --cov-report=html

# Run specific test file
pytest tests/test_data_generation.py

# Run specific test
pytest tests/test_data_generation.py::test_make_circles
```

### Writing Tests

```python
import pytest
import numpy as np
from qbiocode.data_generation import make_circles

def test_make_circles_basic():
    """Test basic circle generation."""
    X, y = make_circles(n_samples=100, noise=0.1)
    
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert set(y) == {0, 1}

def test_make_circles_invalid_input():
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        make_circles(n_samples=-10)
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Include integration tests for workflows
- Test on multiple Python versions (3.8, 3.9, 3.10, 3.11)

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: Check docs first for answers

### Recognition

Contributors are recognized in:
- Release notes
- CITATION.cff file
- Project documentation

### Stay Connected

- Watch the repository for updates
- Star the project if you find it useful
- Share your work using QBioCode

## Questions?

If you have questions about contributing, feel free to:
- Open a discussion on GitHub
- Create an issue with the "question" label
- Reach out to the maintainers

---

**Thank you for contributing to QBioCode!** Your efforts help advance quantum for biological and healthcare applications. 🚀
