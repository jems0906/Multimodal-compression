"""TVM (Apache TVM) relay-IR export and benchmark utilities.

TVM is an optional dependency.  All functions degrade gracefully when the
``tvm`` package is not installed, returning a status dict that the pipeline
can include in its metrics without crashing.

Typical usage
-------------
    from src.compression.tvm_export import export_with_tvm, benchmark_tvm

    tvm_info = export_with_tvm(model, export_path="experiments/tvm", target="llvm")
    metrics  = benchmark_tvm(tvm_info, warmup_runs=3, benchmark_runs=20)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import tvm  # type: ignore[reportMissingImports]
    from tvm import relay  # type: ignore[reportMissingImports]
    from tvm.contrib import graph_executor  # type: ignore[reportMissingImports]

    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_example_inputs_numpy(model, batch_size: int = 1) -> dict[str, np.ndarray]:
    """Return processor-generated inputs as NumPy arrays for TVM tracing."""
    sample_rate = getattr(model.processor, "sampling_rate", None)
    if sample_rate is None and hasattr(model.processor, "feature_extractor"):
        sample_rate = getattr(model.processor.feature_extractor, "sampling_rate", 16000)
    sample_rate = int(sample_rate or 16000)

    duration = getattr(model.processor, "max_length_s", None)
    if duration is None and hasattr(model.processor, "feature_extractor"):
        duration = getattr(model.processor.feature_extractor, "max_length_s", 10)
    duration = max(1, int(round(float(duration or 10))))

    waveforms = [np.zeros(sample_rate * duration, dtype=np.float32) for _ in range(batch_size)]
    processed = model.processor(audio=waveforms, return_tensors="pt", sampling_rate=sample_rate)
    if "is_longer" not in processed:
        processed["is_longer"] = torch.zeros((batch_size,), dtype=torch.bool)
    return {k: v.numpy() for k, v in processed.items() if torch.is_tensor(v)}


def _torch_to_relay(model: Any, example_inputs: dict[str, np.ndarray], target_str: str):
    """Trace the PyTorch model and convert it to TVM Relay IR via ``relay.frontend.from_pytorch``."""
    # Build ordered list matching processor output order
    ordered_inputs = [(k, v) for k, v in example_inputs.items()]
    input_shapes = [(k, v.shape) for k, v in ordered_inputs]
    input_tensors = tuple(torch.tensor(v) for _, v in ordered_inputs)

    # Trace with TorchScript
    nn_module = model.model.eval().cpu()
    with torch.no_grad():
        scripted = torch.jit.trace(nn_module, input_tensors, strict=False)

    mod, params = relay.frontend.from_pytorch(scripted, input_shapes)
    return mod, params, input_shapes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_with_tvm(
    model,
    export_path: str = "experiments/tvm",
    target: str = "llvm",
    opt_level: int = 3,
) -> dict:
    """Compile a multimodal model with TVM and save artefacts.

    Parameters
    ----------
    model:
        A ``MultimodalModel`` wrapper instance.
    export_path:
        Directory where ``lib.so`` (or ``lib.tar`` on non-Linux), ``graph.json``
        and ``params.bin`` artefacts are saved.
    target:
        TVM target string, e.g. ``"llvm"`` (CPU) or ``"cuda"`` (NVIDIA GPU).
    opt_level:
        Relay optimisation level (0–3).

    Returns
    -------
    dict with keys: ``tvm_status``, ``tvm_export_path``, ``tvm_target``,
    ``tvm_opt_level``, and on error ``tvm_error``.
    """
    if not TVM_AVAILABLE:
        return {"tvm_status": "skipped", "tvm_error": "tvm package not installed"}

    try:
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        example_inputs = _build_example_inputs_numpy(model, batch_size=1)
        mod, params, _input_shapes = _torch_to_relay(model, example_inputs, target)

        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(mod, target=target, params=params)

        # Save artefacts
        lib_path = export_dir / "lib.tar"
        graph_path = export_dir / "graph.json"
        params_path = export_dir / "params.bin"

        lib.export_library(str(lib_path))
        with open(graph_path, "w") as f:
            f.write(lib.get_graph_json())
        with open(params_path, "wb") as f:
            f.write(relay.save_param_dict(lib.get_params()))

        return {
            "tvm_status": "success",
            "tvm_export_path": str(export_dir),
            "tvm_lib_path": str(lib_path),
            "tvm_target": target,
            "tvm_opt_level": opt_level,
            # Stash lib handle so benchmark_tvm can reuse without re-compiling
            "_tvm_lib": lib,
            "_tvm_example_inputs": example_inputs,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "tvm_status": "failed",
            "tvm_error": str(exc)[:400],
        }


def benchmark_tvm(
    tvm_info: dict,
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
    target: str = "llvm",
) -> dict:
    """Run inference benchmarks on a previously exported TVM module.

    Parameters
    ----------
    tvm_info:
        The dict returned by :func:`export_with_tvm`.
    warmup_runs / benchmark_runs:
        Number of warm-up and timed iterations.
    target:
        TVM target string used during export.

    Returns
    -------
    dict with keys: ``tvm_status``, ``tvm_latency_ms``,
    ``tvm_throughput_samples_per_sec``, ``tvm_provider``.
    """
    if not TVM_AVAILABLE:
        return {"tvm_status": "skipped", "tvm_error": "tvm package not installed"}

    if tvm_info.get("tvm_status") != "success":
        return {"tvm_status": "skipped", "tvm_error": "export did not succeed"}

    try:
        lib = tvm_info["_tvm_lib"]
        example_inputs = tvm_info["_tvm_example_inputs"]

        dev = tvm.cpu() if "llvm" in target else tvm.cuda()
        module = graph_executor.GraphModule(lib["default"](dev))

        # Set inputs
        for name, arr in example_inputs.items():
            module.set_input(name, tvm.nd.array(arr, dev))

        # Warm-up
        for _ in range(warmup_runs):
            module.run()

        # Timed runs
        latencies = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            module.run()
            latencies.append(time.perf_counter() - start)

        avg_latency = sum(latencies) / max(len(latencies), 1)
        return {
            "tvm_status": "success",
            "tvm_latency_ms": avg_latency * 1000.0,
            "tvm_throughput_samples_per_sec": 1.0 / avg_latency if avg_latency > 0 else 0.0,
            "tvm_provider": target,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "tvm_status": "failed",
            "tvm_error": str(exc)[:400],
        }
