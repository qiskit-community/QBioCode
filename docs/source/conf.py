# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import shutil
import sys

# Anchor every path on this file, not on the cwd. `make html` runs from docs/ but
# `sphinx-build docs/source docs/build/html` runs from the repository root, and
# cwd-relative paths silently resolved to the wrong place in the second case.
_CONF_DIR = os.path.dirname(os.path.abspath(__file__))       # docs/source
_DOCS_DIR = os.path.dirname(_CONF_DIR)                       # docs
_REPO_ROOT = os.path.dirname(_DOCS_DIR)                      # checkout root

sys.path.insert(0, _CONF_DIR)
sys.path.insert(0, _DOCS_DIR)
sys.path.insert(0, _REPO_ROOT)

# Mock imports so autodoc can introspect every module -- including the QuVINE app
# (``qbiocode.apps.quvine``) -- in an environment that has only the [docs] extra.
#
# Everything here is either a member of the optional [quvine] extra, which a docs
# build has no reason to install, or a package whose import is unreliable in CI.
# Base runtime dependencies are deliberately NOT mocked: `pip install -e ".[docs]"`
# installs them, and mocking a real dependency hides genuine import errors.
# `tensorflow` used to be listed and was removed: nothing in qbiocode, the
# tutorials, or the docs imports TensorFlow or Keras.
autodoc_mock_imports = [
    # [quvine] extra
    'gensim',
    'hiperwalk',
    'node2vec',
    'torch_geometric',
    'community',        # python-louvain
    'ripser',
    # omegaconf backs QuVINE's config loading (api/config, api/core, api/sgns,
    # main, pipeline, utils/io), so autodoc cannot import those modules without it.
    'omegaconf',
    # unreliable wheels in CI
    'xgboost',
]


project = 'qbiocode'
copyright = '2025 IBM Research' #, Bryan Raubenolt, Aritra Bose, Kahn Rhrissorrakrai, Filippo Utro, Akhil Mohan, Daniel Blankenberg, Laxmi Parida'
author = 'Bryan Raubenolt, Aritra Bose, Kahn Rhrissorrakrai, Filippo Utro, Akhil Mohan, Daniel Blankenberg, Laxmi Parida'


def _package_version():
    """Read ``__version__`` out of qbiocode/version.py.

    Parsed rather than imported: importing qbiocode pulls in qiskit, torch and
    umap, which a docs-only environment may not have, and a hardcoded literal
    here silently disagreed with the package for several releases (conf.py said
    0.0.1 while qbiocode/version.py said 0.1.0).
    """
    import re

    version_file = os.path.join(_REPO_ROOT, "qbiocode", "version.py")
    try:
        with open(version_file, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise RuntimeError(
            f"Cannot read {version_file} to determine the documented version. "
            "conf.py expects to live at docs/source/conf.py inside the checkout."
        ) from exc
    match = re.search(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", text, re.M)
    if match is None:
        raise RuntimeError(
            f"No __version__ assignment found in {version_file}; cannot set the "
            "documented release."
        )
    return match.group(1)


release = _package_version()
version = release

# Documentation note
html_show_sphinx = True
html_show_copyright = True

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
     "nbsphinx",
   "myst_parser",
   "sphinx_design",
  # "myst_nb"
]

templates_path = ['_templates']
# workshops/_tutorial.rst is a copy-me skeleton full of {{placeholder}} braces, not
# a page: un-excluding it publishes "{{Workshop Title}}" and, since no toctree
# references it, adds an "isn't included in any toctree" warning as well.
exclude_patterns = ["build", "Thumbs.db", ".DS_Store", "workshops/_tutorial.rst"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_css_files = ['custom.css']  # Add custom CSS
nbsphinx_execute = 'never' # O 'auto', 'always'. 'never'

# nbsphinx configuration for better notebook rendering
nbsphinx_prolog = """
.. raw:: html

    <style>
        /* Improve notebook cell spacing */
        div.nbinput.container,
        div.nboutput.container {
            margin-bottom: 1em;
        }
        
        /* Better code cell styling */
        div.highlight-ipython3 {
            margin: 0.5em 0;
        }
        
        /* Improve markdown cell rendering */
        .nbinput + .nboutput {
            margin-top: 0.5em;
        }
    </style>
"""

# Prompt for notebook execution
nbsphinx_prompt_width = "0"


# Impostazioni di MyST-Parser
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# nbsphinx automatically handles .ipynb files when included in extensions

# -- tutorial/ -> docs/source/tutorials/ -------------------------------------
#
# ``docs/source/tutorials/`` is generated, not authored: it is a filtered copy of
# the repository's top-level ``tutorial/`` tree, made at build time by
# ``_sync_tutorials`` below and gitignored.
#
# It used to be a hand-maintained second copy of the same notebooks, and the two
# drifted exactly as you would expect. ``sc_binary_qprofiler.ipynb`` grew an
# ADT/protein modality on the docs side only; ``qsage.ipynb``'s
# ``data/qprofiler_benchmarks.csv`` fixture existed only on the docs side, so the
# ``tutorial/`` copy of that notebook could not run at all; and two notebooks
# existed under docs and nowhere else. ``tutorial/`` is now the single source of
# truth and this hook derives the docs tree from it.

#: Extensions worth copying. A notebook needs its own ``.ipynb``, the helper
#: module ``QEnsemble/QEnsemble_example_blobs.ipynb`` imports, the config files
#: readers are pointed at, and the figures a page embeds.
_TUTORIAL_SUFFIXES = frozenset({".ipynb", ".py", ".yaml", ".json", ".png", ".pdf"})

#: Directories never copied. Every one is either notebook *output* or an input
#: fixture that the build has no use for -- see ``_sync_tutorials``' docstring.
_TUTORIAL_SKIP_DIRS = frozenset(
    {
        "data",
        "datasets",
        "experiments",
        "outputs",
        "OV",
        "pqk_projections",
        "qpl_projections",
        "performance_summary_and_spearman_correlation_plots",
        "__pycache__",
        ".ipynb_checkpoints",
    }
)

#: Written into the generated tree so the hook can prove a directory is its own
#: before deleting it.
_TUTORIAL_MARKER = ".generated"


def _sync_tutorials(app=None):
    """Regenerate ``docs/source/tutorials/`` from ``tutorial/``.

    Two exclusions are load-bearing rather than merely tidy:

    * **No ``.md``.** ``tutorial/QEnsemble/README.md`` would be picked up by
      myst_parser as a source document, reachable from no toctree, and
      "document isn't included in any toctree" is a warning -- which ``-W`` in
      ``docs/Makefile`` turns into a failed build.
    * **No data directories.** ``nbsphinx_execute = 'never'``, so no notebook
      runs during the build and nbsphinx offers only the ``.ipynb`` itself for
      download. The fixtures under ``tutorial/**/data/``,
      ``tutorial/QuVINE/datasets/`` and ``tutorial/QEnsemble/experiments/`` are
      ~20 MB that nothing in the built site would ever read.

    The destination is deleted and rebuilt so a notebook renamed or removed
    upstream cannot leave a stale page behind. That deletion is guarded: a
    directory without the ``.generated`` marker is assumed to be hand-authored
    and the build fails rather than removing it. ``copy2`` preserves mtimes, so
    Sphinx's incremental rebuild still works across runs.
    """
    from sphinx.util import logging as sphinx_logging

    logger = sphinx_logging.getLogger(__name__)

    source = os.path.join(_REPO_ROOT, "tutorial")
    dest = os.path.join(_CONF_DIR, "tutorials")

    if not os.path.isdir(source):
        raise RuntimeError(
            f"{source} does not exist, so the tutorial pages cannot be generated "
            "and every tutorials/* toctree entry would fail to resolve. The "
            "notebooks ship in the sdist (MANIFEST.in: recursive-include tutorial "
            "*.ipynb); build the docs from a full checkout or sdist."
        )

    if os.path.isdir(dest):
        if not os.path.exists(os.path.join(dest, _TUTORIAL_MARKER)):
            raise RuntimeError(
                f"{dest} exists but carries no {_TUTORIAL_MARKER} marker, so it "
                "was not generated by this hook and will not be deleted. It is "
                "supposed to be a derived copy of tutorial/ -- move anything you "
                "want to keep into tutorial/ and remove the directory."
            )
        shutil.rmtree(dest)

    copied = 0
    for current, dirnames, filenames in os.walk(source):
        dirnames[:] = sorted(d for d in dirnames if d not in _TUTORIAL_SKIP_DIRS)
        relative = os.path.relpath(current, source)
        target_dir = dest if relative == os.curdir else os.path.join(dest, relative)
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1] not in _TUTORIAL_SUFFIXES:
                continue
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(os.path.join(current, filename), os.path.join(target_dir, filename))
            copied += 1

    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, _TUTORIAL_MARKER), "w", encoding="utf-8") as handle:
        handle.write(
            "Generated by _sync_tutorials() in docs/source/conf.py from the\n"
            "repository's tutorial/ directory. Do not edit and do not commit;\n"
            "edit the originals under tutorial/ instead.\n"
        )

    logger.info("[tutorials] copied %d files from tutorial/ into %s", copied, dest)


def run_apidoc(app):
    """Regenerate docs/source/api/ from the package tree with better_apidoc.

    Two things used to go wrong here:

    * Every path was cwd-relative, so this only worked when sphinx-build was
      invoked from ``docs/``. Paths are now anchored on ``conf.py``.
    * A bare ``except Exception`` printed a note to stdout and continued. A
      partial run therefore left a half-regenerated ``api/`` tree behind with no
      failing exit code -- which is how the committed pages ended up a mix of two
      generators' output, missing ``qbiocode.apps`` entirely. Failures are now
      reported through Sphinx's own logger, so ``-W`` turns them into build
      errors.

    A missing better_apidoc is reported at info level rather than as a warning:
    the committed ``api/*.rst`` pages are a complete, checked-in fallback, so a
    build without the tool still produces a full reference -- just one that is
    only as fresh as the last commit.
    """
    from sphinx.util import logging as sphinx_logging

    logger = sphinx_logging.getLogger(__name__)

    try:
        import better_apidoc
    except ImportError:
        logger.info(
            "better_apidoc is not installed; using the committed docs/source/api "
            "pages as-is. Install it with: pip install \"qbiocode[docs]\""
        )
        return

    try:
        better_apidoc.APP = app
        better_apidoc.main(
            [
                "better-apidoc",
                "-t",
                os.path.join(_DOCS_DIR, "_templates"),
                "--force",
                "--no-toc",
                "--separate",
                "-o",
                os.path.join(_CONF_DIR, "api"),
                os.path.join(_REPO_ROOT, "qbiocode"),
            ]
        )
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        logger.warning(
            "better_apidoc failed after possibly rewriting part of "
            "docs/source/api/: %s: %s. The API reference may be incomplete; "
            "`git checkout docs/source/api` to restore the committed pages.",
            type(exc).__name__,
            exc,
        )


# -- Extension configuration -------------------------------------------------
add_module_names = False


napoleon_google_docstring = True
napoleon_include_init_with_doc = True
# Render a Google-style ``Attributes:`` section as :ivar: fields inside the
# class description instead of as standalone attribute descriptions.  Without
# this, a dataclass whose docstring lists its fields has each field described
# twice -- once from the docstring, once from ``:undoc-members:`` walking the
# annotations -- which Sphinx reports as "duplicate object description".
napoleon_use_ivar = True

# The docs build is warning-free, and ``-W`` in the Makefile keeps it that way.
# One warning is not ours to fix: sphinx-autodoc-typehints resolves annotations
# by importing the modules they name, and pydantic's dataclass internals annotate
# against ``_typeshed``, which is a typing-only stub that never exists at
# runtime. The failure is reported once, changes nothing in the output, and
# cannot be fixed from this repository -- so exactly that subtype is suppressed,
# and nothing broader.
suppress_warnings = ["sphinx_autodoc_typehints.guarded_import"]

coverage_ignore_modules = []
coverage_ignore_functions = []
coverage_ignore_classes = []


myst_enable_extensions = [
    "colon_fence",  # Ensures ::: blocks work
  #  "linkify",
    "strikethrough",
    "tasklist",
    # ... any other extensions you want
]


coverage_show_missing_items = True
html_theme = 'pydata_sphinx_theme' #'sphinx_rtd_theme' # 'furo'

html_show_sourcelink = False

html_logo = "_static/QBioCode_logo.png"
if os.path.exists(os.path.join(os.path.dirname(__file__), "_static", "favicon.ico")):
    html_favicon = "_static/favicon.ico"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/qiskit-community/QBioCode",
            "icon": "fab fa-github",
            "type": "fontawesome",
        }
    ],
    "show_prev_next": True,  # Enable prev/next navigation
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],  # Add navigation to center
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "header_links_before_dropdown": 8,
    "navigation_depth": 3,  # Increased depth for better navigation
    "show_toc_level": 2,  # Show 2 levels in TOC
    "collapse_navigation": False,  # Keep navigation expanded
    "navigation_with_keys": True,  # Enable keyboard navigation
}

html_context = {
    "default_mode": "light",
}

# Footer text
rst_epilog = """
.. |ai_note| replace:: *Portions of this documentation were generated with AI assistance.*
"""

# Enable sidebars for better navigation
html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

def setup(app):
    # Both generate source files, so both must run before Sphinx reads the
    # source tree. They are independent of each other.
    app.connect("builder-inited", _sync_tutorials)
    app.connect("builder-inited", run_apidoc)
