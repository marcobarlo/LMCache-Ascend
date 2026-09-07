# SPDX-License-Identifier: Apache-2.0
"""Source-level checks for the MP wiring failures seen in live dual-pass."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYBIND = REPO / "csrc" / "pybind.cpp"
INIT = REPO / "lmcache_ascend" / "__init__.py"


def test_pybind_page_buffer_shape_desc_is_dynamic_attr() -> None:
    text = PYBIND.read_text()
    assert "PageBufferShapeDesc" in text
    assert "py::dynamic_attr()" in text


def test_pybind_does_not_release_gil_before_block_transfer() -> None:
    """TORCH_CHECK after gil_scoped_release aborted the lmcache server."""
    text = PYBIND.read_text()
    start = text.index('m.def(\n      "multi_layer_block_kv_transfer"')
    end = text.index("py::class_<StagingCopy>", start)
    lambda_body = text[start:end]
    assert "gil_scoped_release" not in lambda_body


def test_pybind_does_not_isinstance_page_buffer_shape_desc() -> None:
    """Cross-module py::isinstance<PageBufferShapeDesc> SIGSEGVs on affinity workers."""
    text = PYBIND.read_text()
    start = text.index('m.def(\n      "multi_layer_block_kv_transfer"')
    end = text.index("py::class_<StagingCopy>", start)
    lambda_body = text[start:end]
    assert "py::isinstance<PageBufferShapeDesc>" not in lambda_body


def test_dispatcher_passes_converted_paged_arg() -> None:
    text = INIT.read_text()
    assert "_paged_arg_to_ptr_tensor" in text
    assert "return _native_block(\n                paged," in text
    assert "execute_object_group_transfer = None" not in text
    assert "_HAS_NATIVE_OBJECT_GROUP_TRANSFER = False" not in text
    assert "PageBufferShapeDesc as _NativeShapeDesc" in text
    assert "_fmt_13" in text
    assert "NL_X_NB_BS_NH_CS" in text
    assert "(_fmt_13, _fmt_16, _fmt_17)" in text


def test_pybind_exports_execute_object_group_transfer() -> None:
    text = PYBIND.read_text()
    assert '"execute_object_group_transfer"' in text
    assert "static_cast<TransferDirection>(direction)" in text
    start = text.index('"execute_object_group_transfer"')
    end = text.index('py::arg("batch_steps")', start)
    assert "gil_scoped_release" not in text[start:end]


def test_plan_op_does_not_reenter_direct_entry() -> None:
    """Per-group multi_layer_block_kv_transfer would Run() again inside the plan."""
    text = (REPO / "csrc" / "mp_mem_kernels.cpp").read_text()
    start = text.index("void execute_object_group_transfer")
    end = text.index("void lmcache_memcpy_async", start)
    body = text[start:end]
    assert "multi_layer_block_kv_transfer(" not in body
    assert body.count("cmd.Run()") == 1
    assert "launch_block_transfer_objects" in body
