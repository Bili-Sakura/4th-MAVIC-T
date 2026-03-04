# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for packaging metadata and version accessibility."""

import re

import pytest


def test_version_string():
    """__version__ is a valid semver string."""
    from src import __version__

    assert isinstance(__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", __version__), (
        f"Version {__version__!r} does not look like semver"
    )


def test_public_submodules_importable():
    """The four public sub-packages are importable when torch is available."""
    pytest.importorskip("torch")
    import src.models
    import src.pipelines
    import src.schedulers
    import src.utils
