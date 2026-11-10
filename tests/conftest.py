"""Shared test setup: mark tests that need the internet or a GPU, and skip them in CI.

GitHub Actions sets the CI environment variable. There the tests run on a CPU with no
Hugging Face downloads, so tests that need either are skipped instead of failing.
"""

import os

import pytest

# Fixtures that download Qwen2.5's tokenizer from the Hugging Face Hub. Any test using
# one of them, directly or through another fixture, needs the internet.
INTERNET_FIXTURES = {"tokenizer", "qwen_tokenizer", "tiny_run"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    in_ci = bool(os.environ.get("CI"))
    for item in items:
        if INTERNET_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.internet)
        if not in_ci:
            continue
        for marker in ("internet", "gpu"):
            if item.get_closest_marker(marker):
                item.add_marker(pytest.mark.skip(reason=f"needs {marker}; skipped in CI"))
