"""Smoke test: every module in the package can be imported."""

import importlib

import pytest

MODULES = ["data", "verify", "formatting", "train", "evaluate"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(f"tooltuned.{name}")
