"""Unit Tests for PixelLab 50-Filter Library
Verifies shape preservation, uint8 dtype compliance, and trace engine generation across all 50 filters.
"""

import pytest
import numpy as np
from filters import FILTER_REGISTRY, get_filters_metadata


@pytest.fixture
def sample_rgba_image():
    """Create a deterministic synthetic RGBA test image (48x48x4)."""
    np.random.seed(42)
    img = np.random.randint(0, 256, (48, 48, 4), dtype=np.uint8)
    img[:, :, 3] = 255  # fully opaque
    return img


def test_registry_has_50_filters():
    assert len(FILTER_REGISTRY) == 50, f"Expected exactly 50 filters, found {len(FILTER_REGISTRY)}"


@pytest.mark.parametrize("filter_id", list(FILTER_REGISTRY.keys()))
def test_filter_output_shape_and_dtype(filter_id, sample_rgba_image):
    entry = FILTER_REGISTRY[filter_id]
    fn = entry["fn"]
    params = {p["id"]: p["default"] for p in entry["params"]}

    # Process image through NumPy implementation
    result = fn(sample_rgba_image, **params)

    # Check return type and shape
    assert isinstance(result, np.ndarray), f"Filter {filter_id} did not return ndarray"
    assert result.dtype == np.uint8, f"Filter {filter_id} dtype is {result.dtype}, expected uint8"
    assert result.shape == sample_rgba_image.shape, (
        f"Filter {filter_id} shape {result.shape} does not match input {sample_rgba_image.shape}"
    )


@pytest.mark.parametrize("filter_id", list(FILTER_REGISTRY.keys()))
def test_filter_trace_fn(filter_id):
    entry = FILTER_REGISTRY[filter_id]
    trace_fn = entry.get("trace_fn")
    assert trace_fn is not None, f"Filter {filter_id} missing trace_fn"

    params = {p["id"]: p["default"] for p in entry["params"]}
    trace_data = trace_fn(200, 150, 80, 255, params)

    assert "formula" in trace_data, f"Filter {filter_id} trace missing 'formula'"
    assert "steps" in trace_data, f"Filter {filter_id} trace missing 'steps'"
    assert isinstance(trace_data["steps"], list) and len(trace_data["steps"]) > 0


def test_metadata_extraction():
    meta = get_filters_metadata()
    assert len(meta) == 50
    for fid, item in meta.items():
        assert "source_code" in item and len(item["source_code"]) > 0, f"Filter {fid} missing source_code"
        assert "description" in item and len(item["description"]) > 0, f"Filter {fid} missing description"
        assert "math" in item and len(item["math"]) > 0, f"Filter {fid} missing math"
