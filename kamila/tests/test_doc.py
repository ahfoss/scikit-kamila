"""Unit test for Sphinx documentation build."""

from pathlib import Path

import pytest


def test_sphinx_documentation_build(tmp_path):
    """Verify that Sphinx documentation builds successfully and outputs HTML pages."""
    sphinx_cmd_build = pytest.importorskip("sphinx.cmd.build")

    doc_dir = Path(__file__).resolve().parents[2] / "doc"
    out_dir = tmp_path / "html"

    ret = sphinx_cmd_build.build_main(
        [
            "-b",
            "html",
            str(doc_dir),
            str(out_dir),
        ]
    )

    assert ret == 0, f"Sphinx build failed with return code {ret}"
    assert (out_dir / "index.html").exists(), "index.html was not generated"
    assert (out_dir / "api.html").exists(), "api.html was not generated"
    assert (out_dir / "user_guide.html").exists(), "user_guide.html was not generated"
    assert (out_dir / "quick_start.html").exists(), "quick_start.html was not generated"
    assert (
        out_dir / "auto_examples" / "index.html"
    ).exists(), "auto_examples/index.html was not generated"
    assert (
        out_dir / "auto_examples" / "plot_kamila.html"
    ).exists(), "plot_kamila.html was not generated"
