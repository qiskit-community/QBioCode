# Copyright 2026, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""What library plotting code must not do to the process that imported it.

A plotting function is a library call like any other, and three habits make it
behave otherwise:

1. **Mutating global state at import time.** ``import qbiocode`` used to assign 27
   entries into ``plt.rcParams``, so every unrelated figure the caller drew
   afterwards came out in Arial with no top spine and a 600-dpi savefig default,
   with nothing in the call stack to attribute it to.
2. **Blocking.** An unconditional ``plt.show()`` hangs a batch run under a GUI
   backend and warns three times per call under ``Agg``.
3. **Leaking figures.** ``plt.close()`` closes whatever figure is *current*, which
   is not necessarily the one just drawn. QProfiler plots once per metric per
   embedding per iteration, so a leak here is unbounded.

A fourth is silent output loss: deriving one output path from another with
``re.sub(".pdf", ...)`` collapsed three figures onto one file for any format but
PDF, and mangled paths containing ``?pdf``.
"""

import importlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The two first-party plotting modules are imported directly. Guarding a module
# of our own with ``pytest.importorskip`` means an ImportError inside it -- a
# real regression -- silently turns its tests into skips instead of failures.
# Both need only matplotlib, numpy and scikit-learn, all mandatory in
# requirements-base.txt, so there is no install where they are legitimately
# absent.
# ``import_module``, not ``from X import Y``: ``qbiocode.apps.sage`` re-exports a
# name that shadows its own ``sage`` submodule, so ``from`` binds the wrong
# object depending on what has already been imported.
_analyze = importlib.import_module("qbiocode.apps.quvine.analysis.analyze")
_sage = importlib.import_module("qbiocode.apps.sage.sage")


# ---------------------------------------------------------------------------
# 1. Importing the package must not reconfigure the caller's matplotlib
# ---------------------------------------------------------------------------
def test_importing_qbiocode_leaves_rcparams_alone():
    """Run in a subprocess: this process has already imported qbiocode."""
    code = (
        "import matplotlib;matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "keys=['savefig.dpi','savefig.bbox','font.family','font.size','axes.labelsize',"
        "'axes.spines.top','axes.spines.right','xtick.major.size','grid.linestyle',"
        "'axes.linewidth']\n"
        "before={k:repr(plt.rcParams[k]) for k in keys}\n"
        "import qbiocode\n"
        "changed=[k for k in keys if repr(plt.rcParams[k])!=before[k]]\n"
        "print('CHANGED:'+','.join(changed))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr[-3000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("CHANGED:")][0]
    assert line == "CHANGED:", (
        f"importing qbiocode modified {line[len('CHANGED:'):]} in the caller's "
        f"plt.rcParams. Publication styling belongs in a per-figure rc_context."
    )


def test_the_style_is_available_to_opt_into():
    import qbiocode
    from qbiocode.visualization import PUBLICATION_STYLE, publication_style

    style = publication_style()
    assert style["savefig.dpi"] == 600
    assert style == PUBLICATION_STYLE
    assert qbiocode.publication_style() == style

    # A copy, not the module default: mutating what a caller is handed must not
    # change what the next caller gets.
    style["savefig.dpi"] = 72
    style["font.sans-serif"].append("Comic Sans MS")
    assert publication_style()["savefig.dpi"] == 600
    assert "Comic Sans MS" not in publication_style()["font.sans-serif"]


# ---------------------------------------------------------------------------
# 2. Output paths: one file per figure, whatever the extension
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("plots/corr.pdf", ("plots/corr_heatmap.pdf", "plots/corr_noncluster_heatmap.pdf")),
        ("plots/corr.png", ("plots/corr_heatmap.png", "plots/corr_noncluster_heatmap.png")),
        # '.' was an unescaped regex wildcard, so '_pdf' matched and was rewritten.
        ("out/spdf_x.pdf", ("out/spdf_x_heatmap.pdf", "out/spdf_x_noncluster_heatmap.pdf")),
        ("corr", ("corr_heatmap", "corr_noncluster_heatmap")),
    ],
)
def test_derived_paths_never_collide_with_the_original(path, expected):
    from qbiocode.visualization.visualize_correlation import _derive_path

    derived = (_derive_path(path, "_heatmap"), _derive_path(path, "_noncluster_heatmap"))
    assert derived == expected
    assert path not in derived, "a derived path equal to the original overwrites it"
    assert derived[0] != derived[1]


def _correlation_frame():
    rng = np.random.default_rng(0)
    rows = [
        {
            "model_embed_datatype": model,
            "feature": feature,
            "correlation": rng.uniform(-1, 1),
            "median_metric": rng.uniform(0.4, 0.95),
            "metric": "f1_score",
            "p_value": rng.uniform(0, 0.2),
        }
        for model in ("RF_pca_sc", "SVC_pca_sc", "QSVC_pca_sc", "PQK_pca_sc")
        for feature in ("n_features", "class_balance", "intrinsic_dim", "noise")
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("ext", [".pdf", ".png"])
def test_plot_results_correlation_writes_one_file_per_figure(tmp_path, ext):
    """The regression that matters: a .png path used to produce a single file."""
    from qbiocode import plot_results_correlation

    target = tmp_path / f"corr{ext}"
    plot_results_correlation(_correlation_frame(), save_file_path=str(target))

    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == [
        f"corr{ext}",
        f"corr_heatmap{ext}",
        f"corr_noncluster_heatmap{ext}",
    ], written
    assert all((tmp_path / name).stat().st_size > 0 for name in written)


def test_plot_results_correlation_creates_a_missing_output_directory(tmp_path):
    from qbiocode import plot_results_correlation

    target = tmp_path / "nested" / "deeper" / "corr.png"
    plot_results_correlation(_correlation_frame(), save_file_path=str(target))
    assert target.exists()


# ---------------------------------------------------------------------------
# 3. No blocking, no leaking, no global mutation per call
# ---------------------------------------------------------------------------
def test_plot_results_correlation_neither_shows_nor_leaks(tmp_path, monkeypatch):
    from qbiocode import plot_results_correlation

    def explode():  # pragma: no cover - must never run
        raise AssertionError(
            "plt.show() was called under a non-interactive backend; library code "
            "must not attempt to display, it warns or blocks depending on backend."
        )

    monkeypatch.setattr(plt, "show", explode)

    plt.close("all")
    before_rc = dict(plt.rcParams)
    figs = plot_results_correlation(
        _correlation_frame(), save_file_path=str(tmp_path / "corr.pdf"), show_plots=True
    )

    assert plt.get_fignums() == [], "figures were left open in pyplot's manager"
    changed = [k for k in before_rc if repr(before_rc[k]) != repr(plt.rcParams[k])]
    assert changed == [], f"the call left {changed} modified in plt.rcParams"

    # The returned figures survive being closed, which is what makes closing safe.
    assert figs._fields == ("scatter", "scatter_ax", "clustered_heatmap", "ordered_heatmap")
    reuse = tmp_path / "reuse.png"
    figs.scatter.savefig(reuse)
    assert reuse.stat().st_size > 0


def test_the_style_applies_during_the_call_even_though_it_is_gone_after(tmp_path):
    """Restoring rcParams must not mean the styling was never applied."""
    from qbiocode import plot_results_correlation

    figs = plot_results_correlation(
        _correlation_frame(), save_file_path=str(tmp_path / "corr.pdf")
    )
    assert figs.scatter_ax.spines["top"].get_visible() is False, (
        "PUBLICATION_STYLE sets axes.spines.top=False; if the figure shows a top "
        "spine the rc_context did not cover figure creation."
    )


def test_rcparams_are_restored_when_plotting_raises():
    """The exception path is why rc_context lives in the public wrapper."""
    from qbiocode import plot_results_correlation

    before = dict(plt.rcParams)
    with pytest.raises(Exception):
        # A frame with none of the required columns cannot be plotted.
        plot_results_correlation(pd.DataFrame({"nothing": [1, 2, 3]}))
    changed = [k for k in before if repr(before[k]) != repr(plt.rcParams[k])]
    assert changed == [], (
        f"a failed plot left {changed} modified in plt.rcParams -- the style must be "
        f"restored on the exception path too."
    )


# ---------------------------------------------------------------------------
# 4. The same rules in the two other modules that plot
# ---------------------------------------------------------------------------
def test_quvine_spectrum_figures_are_written_where_asked(tmp_path):
    analyze = _analyze

    plt.close("all")
    embeddings = [np.random.default_rng(i).normal(size=(40, 10)) for i in range(3)]
    outdir = tmp_path / "spectra"
    ranks = analyze.spectral_info(
        embeddings, ["a", "b", "c"], plot_flag=True, outdir=str(outdir)
    )

    assert set(ranks) == {"a", "b", "c"}
    assert all(np.isfinite(v) for v in ranks.values())
    assert sorted(p.name for p in outdir.iterdir()) == [
        "log_normalized_spectrum.png",
        "log_spectrum.png",
        "loglog_spectrum.png",
    ]
    # Previously these three went to the process's cwd under fixed names.
    assert plt.get_fignums() == []


def test_plot_singular_values_saves_before_it_shows(tmp_path, monkeypatch):
    analyze = _analyze

    order = []
    monkeypatch.setattr(analyze, "_can_show", lambda: True)
    monkeypatch.setattr(plt, "show", lambda *a, **k: order.append("show"))

    target = tmp_path / "sv.png"
    original_savefig = analyze.plt.Figure.savefig

    def tracking_savefig(self, *args, **kwargs):
        order.append("save")
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(analyze.plt.Figure, "savefig", tracking_savefig)
    analyze.plot_singular_values(np.array([5.0, 3.0, 1.0, 0.5]), filename=str(target), show=True)

    assert target.exists()
    assert order == ["save", "show"], (
        f"expected save-then-show, got {order}. Showing first means the file is not "
        f"written until a human closes the window."
    )


def test_qsage_plot_results_honours_the_extension_it_was_given(tmp_path):
    sage_mod = _sage

    sage = object.__new__(sage_mod.QuantumSage)
    sage._available_metrics = ["f1_score"]
    sage._available_models = ["random_forest", "mlp"]
    rng = np.random.default_rng(0)
    sage._results_subsages = {
        "f1_score": {
            model: {
                "mae": 0.1,
                "mse": 0.02,
                "rmse": 0.14,
                "r2": 0.8,
                "preds": rng.uniform(size=15),
                "y_test": rng.uniform(size=15),
            }
            for model in sage._available_models
        }
    }

    plt.close("all")
    figures = sage.plot_results(saveFile=str(tmp_path / "qsage.png"))

    assert len(figures) == 2
    assert plt.get_fignums() == []
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "qsage_f1_score_barplot.png",
        "qsage_f1_score_scatterplot.png",
    ], "a .png request used to be written as .pdf with .png inside the filename"


def test_qsage_plot_results_reports_no_results_without_printing(capsys, caplog):
    sage_mod = _sage

    sage = object.__new__(sage_mod.QuantumSage)
    sage._available_metrics = []
    sage._available_models = []
    sage._results_subsages = {}

    with caplog.at_level("WARNING", logger=sage_mod.__name__):
        assert sage.plot_results() == []
    assert "Train the QSages first" in caplog.text
    assert capsys.readouterr().out == "", "library code must log, not print"


# ---------------------------------------------------------------------------
# 5. Structural: no module under qbiocode may style matplotlib at import time
# ---------------------------------------------------------------------------
def test_no_module_mutates_global_matplotlib_state_at_import():
    """A grep-style guard, so this cannot creep back into a new module."""
    import ast

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package = os.path.join(root, "qbiocode")
    forbidden = {"use", "rc", "rcdefaults", "set_theme", "set_style", "set"}
    offenders = []

    for dirpath, dirnames, filenames in os.walk(package):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=path)
            # Module level only: a function body is not executed at import time.
            for node in tree.body:
                for sub in ast.walk(node) if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ) else []:
                    rel = os.path.relpath(path, root)
                    if isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store):
                        text = ast.unparse(sub.value)
                        if text.endswith("rcParams"):
                            offenders.append(f"{rel}:{sub.lineno} assigns into {text}")
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                        if sub.func.attr in forbidden and ast.unparse(sub.func.value) in (
                            "matplotlib", "mpl", "plt", "sns", "seaborn"
                        ):
                            offenders.append(
                                f"{rel}:{sub.lineno} calls {ast.unparse(sub.func)}()"
                            )

    assert offenders == [], (
        "these run at import time and change matplotlib for the whole process:\n  "
        + "\n  ".join(offenders)
        + "\nApply styling per figure with plt.rc_context instead."
    )
