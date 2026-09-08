# Conda Package Submission Guide

This guide explains how to submit QBioCode to conda-forge and bioconda for distribution via conda.

## Overview

QBioCode will be available through two conda channels:
- **conda-forge**: General-purpose conda channel (recommended for most users)
- **bioconda**: Specialized channel for bioinformatics packages

## Prerequisites

Before submitting, ensure:
- ✅ Package is published on PyPI (https://pypi.org/project/qbiocode/)
- ✅ GitHub repository is public (https://github.com/qiskit-community/QBioCode)
- ✅ Package has a proper LICENSE file
- ✅ Package has comprehensive documentation
- ✅ All tests pass in CI/CD

## Part 1: Submit to conda-forge

### Step 1: Fork the staged-recipes repository

1. Go to https://github.com/conda-forge/staged-recipes
2. Click "Fork" to create your own copy
3. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/staged-recipes.git
   cd staged-recipes
   ```

### Step 2: Create a new recipe branch

```bash
git checkout -b qbiocode-recipe
```

### Step 3: Copy the recipe

```bash
# Create recipe directory
mkdir -p recipes/qbiocode

# Copy the meta.yaml from this repository
cp /path/to/QBioCode/conda-recipe/meta.yaml recipes/qbiocode/
```

### Step 4: Get the SHA256 hash

After publishing version 0.1.0 to PyPI, get the SHA256 hash:

```bash
# Download the source distribution
pip download --no-deps --no-binary :all: qbiocode==0.1.0

# Calculate SHA256
shasum -a 256 qbiocode-0.1.0.tar.gz
```

Update the `sha256` field in `recipes/qbiocode/meta.yaml` with this hash.

### Step 5: Test the recipe locally (optional but recommended)

```bash
# Install conda-build
conda install conda-build

# Build the package
conda build recipes/qbiocode

# Test the package
conda create -n test-qbiocode qbiocode --use-local
conda activate test-qbiocode
python -c "import qbiocode; print(qbiocode.__version__)"
qprofiler --help
```

### Step 6: Commit and push

```bash
git add recipes/qbiocode/meta.yaml
git commit -m "Add qbiocode recipe"
git push origin qbiocode-recipe
```

### Step 7: Create Pull Request

1. Go to https://github.com/conda-forge/staged-recipes
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:
   - Title: "Add qbiocode"
   - Description: Brief description of the package
   - Checklist: Ensure all items are checked

### Step 8: Address Review Comments

The conda-forge team will review your PR. Common requests:
- Fix dependency versions
- Add missing dependencies
- Update recipe-maintainers list
- Fix test commands

### Step 9: Merge and Feedstock Creation

Once approved:
1. PR will be merged
2. A new feedstock repository will be created: `qbiocode-feedstock`
3. You'll be added as a maintainer
4. Package will be built and published to conda-forge

**Timeline**: Usually 1-2 weeks from submission to availability

## Part 2: Submit to bioconda

Bioconda is specifically for bioinformatics packages and has additional requirements.

### Step 1: Fork the bioconda-recipes repository

1. Go to https://github.com/bioconda/bioconda-recipes
2. Click "Fork"
3. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/bioconda-recipes.git
   cd bioconda-recipes
   ```

### Step 2: Create a new recipe branch

```bash
git checkout -b qbiocode
```

### Step 3: Create the recipe

```bash
# Create recipe directory
mkdir -p recipes/qbiocode

# Copy and adapt the meta.yaml
cp /path/to/QBioCode/conda-recipe/meta.yaml recipes/qbiocode/
```

### Step 4: Adapt for bioconda

Bioconda has specific requirements. Update `meta.yaml`:

```yaml
# Add bioconda-specific metadata
extra:
  identifiers:
    - biotools:qbiocode  # If registered in bio.tools
  additional-platforms:
    - linux-aarch64
    - osx-arm64
```

### Step 5: Test with bioconda-utils

```bash
# Install bioconda-utils
conda install -c conda-forge -c bioconda bioconda-utils

# Test the recipe
bioconda-utils build --packages qbiocode
```

### Step 6: Commit and push

```bash
git add recipes/qbiocode/
git commit -m "Add qbiocode recipe"
git push origin qbiocode
```

### Step 7: Create Pull Request

1. Go to https://github.com/bioconda/bioconda-recipes
2. Create PR from your branch
3. Fill in the template
4. Wait for automated tests to pass

### Step 8: Address Review Comments

Bioconda reviewers will check:
- Recipe follows bioconda guidelines
- All dependencies are available in bioconda or conda-forge
- Tests are comprehensive
- Package is relevant to bioinformatics

### Step 9: Merge

Once approved and tests pass:
1. PR will be merged
2. Package will be built for multiple platforms
3. Published to bioconda channel

**Timeline**: Usually 1-3 weeks from submission to availability

## Post-Submission: Maintaining the Package

### Updating the Package

When releasing a new version:

#### For conda-forge:
1. Fork the feedstock: https://github.com/conda-forge/qbiocode-feedstock
2. Update `recipe/meta.yaml`:
   - Change version number
   - Update SHA256 hash
   - Update dependencies if needed
3. Create PR to the feedstock
4. Automated bot will help with updates

#### For bioconda:
1. Fork bioconda-recipes again
2. Update `recipes/qbiocode/meta.yaml`
3. Create PR
4. Wait for review and merge

### Automated Updates

Both conda-forge and bioconda have bots that can automatically:
- Detect new PyPI releases
- Create PRs with updated recipes
- You just need to review and merge

## Installation After Submission

Once published, users can install via:

```bash
# From conda-forge
conda install -c conda-forge qbiocode

# From bioconda (includes conda-forge)
conda install -c bioconda -c conda-forge qbiocode

# Create environment with qbiocode
conda create -n qbiocode -c conda-forge qbiocode
conda activate qbiocode
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Ensure all dependencies are available in conda-forge or bioconda
2. **Build failures**: Check that the package builds on all platforms (linux, osx, win)
3. **Test failures**: Ensure import tests and CLI commands work correctly
4. **Version conflicts**: Some dependencies may have version constraints

### Getting Help

- **conda-forge**: https://conda-forge.org/docs/maintainer/00_intro.html
- **bioconda**: https://bioconda.github.io/contributor/index.html
- **Gitter chat**: 
  - conda-forge: https://gitter.im/conda-forge/conda-forge.github.io
  - bioconda: https://gitter.im/bioconda/Lobby

## Checklist

Before submitting:
- [ ] Package is on PyPI
- [ ] meta.yaml is complete and tested
- [ ] SHA256 hash is correct
- [ ] All dependencies are specified
- [ ] Recipe maintainers are listed
- [ ] Tests are included
- [ ] Documentation links are correct
- [ ] License is specified

## References

- [conda-forge documentation](https://conda-forge.org/docs/)
- [bioconda documentation](https://bioconda.github.io/)
- [Conda build documentation](https://docs.conda.io/projects/conda-build/en/latest/)
- [Example recipes](https://github.com/conda-forge/staged-recipes/tree/main/recipes)