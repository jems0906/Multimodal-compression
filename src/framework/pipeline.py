from copy import deepcopy
from omegaconf import DictConfig
from pathlib import Path
from datetime import datetime, timezone
import json
import torch
import torch.nn.functional as F
import wandb
from ..models import load_model
from ..compression import compress_model
from ..compression.tvm_export import benchmark_tvm, export_with_tvm
from ..evaluation import benchmark_onnx_runtime, benchmark_pytorch_inference, evaluate_model


class PipelineCancelledError(Exception):
    """Raised when cooperative cancellation is requested during a pipeline stage."""


def _emit_progress(progress_hook, stage: str, status: str, message: str, progress_percent: int | None = None, details: dict | None = None):
    if not progress_hook:
        return

    payload = {
        "stage": stage,
        "status": status,
        "message": message,
    }
    if progress_percent is not None:
        payload["progress_percent"] = int(progress_percent)
    if details:
        payload["details"] = details

    try:
        progress_hook(payload)
    except Exception:
        # Progress hooks are optional telemetry plumbing and must not break
        # the actual pipeline execution.
        pass


def _check_cancel(cancel_check, stage: str, reason: str):
    if cancel_check and cancel_check():
        raise PipelineCancelledError(f"{stage.capitalize()} cancelled: {reason}")


def _resolve_benchmark_model_path(cfg: DictConfig):
    """Prefer finetuned artifact, then compressed artifact, else baseline model."""
    finetuned_path = getattr(cfg.output, "finetuned_model_path", None)
    if finetuned_path and Path(finetuned_path).exists():
        return finetuned_path

    compressed_path = getattr(cfg.output, "compressed_model_path", None)
    if compressed_path and Path(compressed_path).exists():
        return compressed_path

    return None


def _measure_model_size(model) -> dict:
    """Return parameter count and on-disk footprint of a model wrapper."""
    nn_module = model.model
    total_params = sum(p.numel() for p in nn_module.parameters())
    trainable_params = sum(p.numel() for p in nn_module.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in nn_module.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in nn_module.buffers())
    size_mb = (param_bytes + buffer_bytes) / (1024 ** 2)
    return {
        "model_total_params": total_params,
        "model_trainable_params": trainable_params,
        "model_size_mb": round(size_mb, 2),
    }


def _is_wandb_enabled(cfg: DictConfig):
    logging_cfg = getattr(cfg, "logging", None)
    if logging_cfg is None:
        return False
    return bool(getattr(logging_cfg, "use_wandb", False))


def _project_name(cfg: DictConfig):
    project_cfg = getattr(cfg, "project", None)
    if project_cfg is None:
        return "multimodal-compression"
    return str(getattr(project_cfg, "name", "multimodal-compression"))


def _sanitize_stem(value: str):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _write_markdown_report(path: Path, cfg: DictConfig, metrics: dict):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Benchmark Report",
        "",
        f"- Generated: {timestamp}",
        f"- Project: {cfg.project.name}",
        f"- Model: {cfg.model.name}",
        f"- Dataset: {cfg.dataset.name}:{cfg.dataset.split}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]

    for key in sorted(metrics.keys()):
        lines.append(f"| {key} | {metrics[key]} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_benchmark_reports(cfg: DictConfig, metrics: dict):
    benchmark_cfg = getattr(cfg, "benchmark", None)
    reporting_cfg = getattr(benchmark_cfg, "reporting", None) if benchmark_cfg else None

    enabled = bool(getattr(reporting_cfg, "enabled", True))
    if not enabled:
        return {}

    output_dir = Path(getattr(reporting_cfg, "output_dir", "experiments/reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    project_stem = _sanitize_stem(getattr(cfg.project, "name", "benchmark"))
    base_name = f"{project_stem}_{timestamp}"

    json_path = output_dir / f"{base_name}.json"
    markdown_path = output_dir / f"{base_name}.md"

    json_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    _write_markdown_report(markdown_path, cfg, metrics)

    return {
        "report_json_path": str(json_path),
        "report_markdown_path": str(markdown_path),
    }

def run_pipeline(cfg: DictConfig, stage: str, progress_hook=None, cancel_check=None):
    """Run the compression pipeline."""
    wandb_enabled = _is_wandb_enabled(cfg)
    if wandb_enabled:
        wandb.init(project=_project_name(cfg), config=dict(cfg))

    try:
        _emit_progress(progress_hook, stage, "running", f"{stage.capitalize()} started", 0)

        if stage == "analyze":
            _check_cancel(cancel_check, stage, "before model load")
            _emit_progress(progress_hook, stage, "running", "Loading model", 10)
            model = load_model(cfg.model)

            _check_cancel(cancel_check, stage, "before evaluation")
            _emit_progress(progress_hook, stage, "running", "Evaluating baseline", 45)
            metrics = evaluate_model(model, cfg.dataset, cfg.evaluation)
            print(f"Baseline metrics: {metrics}")

            if wandb_enabled:
                wandb.log(metrics)

            _emit_progress(progress_hook, stage, "completed", "Analyze completed", 100, {"metrics": metrics})

        elif stage == "compress":
            _check_cancel(cancel_check, stage, "before model load")
            _emit_progress(progress_hook, stage, "running", "Loading model", 10)
            model = load_model(cfg.model)

            _check_cancel(cancel_check, stage, "before compression")
            _emit_progress(progress_hook, stage, "running", "Applying compression", 45)
            compressed_model = compress_model(model, cfg.compression)

            _check_cancel(cancel_check, stage, "before saving compressed model")
            _emit_progress(progress_hook, stage, "running", "Saving compressed model", 85)
            compressed_model.save_pretrained(cfg.output.compressed_model_path)
            _emit_progress(
                progress_hook,
                stage,
                "completed",
                "Compress completed",
                100,
                {"output_path": str(cfg.output.compressed_model_path)},
            )

        elif stage == "finetune":
            _check_cancel(cancel_check, stage, "before loading compressed model")
            _emit_progress(progress_hook, stage, "running", "Loading compressed model", 10)
            compressed_model = load_model(cfg.model, path=cfg.output.compressed_model_path)

            _check_cancel(cancel_check, stage, "before finetuning")
            _emit_progress(progress_hook, stage, "running", "Starting distillation finetune", 20)
            finetuned_model = finetune_model(
                compressed_model,
                cfg.finetune,
                progress_hook=progress_hook,
                cancel_check=cancel_check,
            )

            _check_cancel(cancel_check, stage, "before saving finetuned model")
            _emit_progress(progress_hook, stage, "running", "Saving finetuned model", 95)
            finetuned_model.save_pretrained(cfg.output.finetuned_model_path)
            _emit_progress(
                progress_hook,
                stage,
                "completed",
                "Finetune completed",
                100,
                {"output_path": str(cfg.output.finetuned_model_path)},
            )

        elif stage == "benchmark":
            model_path = _resolve_benchmark_model_path(cfg)

            _check_cancel(cancel_check, stage, "before model load")
            _emit_progress(progress_hook, stage, "running", "Loading benchmark model", 10)
            model = load_model(cfg.model, path=model_path) if model_path else load_model(cfg.model)

            _check_cancel(cancel_check, stage, "before evaluation")
            _emit_progress(progress_hook, stage, "running", "Evaluating model", 30)
            metrics = evaluate_model(model, cfg.dataset, cfg.evaluation)

            # Model size
            metrics.update(_measure_model_size(model))

            if hasattr(cfg, "benchmark"):
                _check_cancel(cancel_check, stage, "before PyTorch benchmark")
                _emit_progress(progress_hook, stage, "running", "Benchmarking PyTorch", 50)
                metrics.update(benchmark_pytorch_inference(model, cfg.benchmark))

                _check_cancel(cancel_check, stage, "before ONNX benchmark")
                _emit_progress(progress_hook, stage, "running", "Benchmarking ONNX Runtime", 68)
                metrics.update(benchmark_onnx_runtime(model, cfg.benchmark))

                # TVM benchmark (optional – gracefully skipped when tvm is absent)
                tvm_cfg = getattr(getattr(cfg, "benchmark", None), "tvm", None)
                if tvm_cfg is not None and getattr(tvm_cfg, "enabled", False):
                    _check_cancel(cancel_check, stage, "before TVM benchmark")
                    _emit_progress(progress_hook, stage, "running", "Benchmarking TVM", 82)
                    tvm_target = str(getattr(tvm_cfg, "target", "llvm"))
                    tvm_export_path = str(getattr(tvm_cfg, "export_path", "experiments/tvm"))
                    tvm_opt_level = int(getattr(tvm_cfg, "opt_level", 3))
                    tvm_warmup = int(getattr(tvm_cfg, "warmup_runs", 5))
                    tvm_bench = int(getattr(tvm_cfg, "benchmark_runs", 20))

                    tvm_info = export_with_tvm(
                        model,
                        export_path=tvm_export_path,
                        target=tvm_target,
                        opt_level=tvm_opt_level,
                    )
                    tvm_metrics = benchmark_tvm(
                        tvm_info,
                        warmup_runs=tvm_warmup,
                        benchmark_runs=tvm_bench,
                        target=tvm_target,
                    )
                    metrics.update(tvm_metrics)

            _check_cancel(cancel_check, stage, "before writing reports")
            _emit_progress(progress_hook, stage, "running", "Writing benchmark reports", 94)
            report_paths = _write_benchmark_reports(cfg, metrics)
            metrics.update(report_paths)

            print(f"Final metrics: {metrics}")
            if wandb_enabled:
                wandb.log(metrics)

            _emit_progress(progress_hook, stage, "completed", "Benchmark completed", 100, {"metrics": metrics})

    except PipelineCancelledError:
        _emit_progress(progress_hook, stage, "cancelled", f"{stage.capitalize()} cancelled", 100)
        raise
    finally:
        if wandb_enabled:
            wandb.finish()

def finetune_model(model, finetune_cfg, progress_hook=None, cancel_check=None):
    """Finetune the compressed model via teacher-student distillation on synthetic inputs."""
    from transformers import AutoModel
    from ..compression.compression import _build_synthetic_inputs, _forward_features

    epochs = int(getattr(finetune_cfg, "epochs", 3))
    lr = float(getattr(finetune_cfg, "lr", 1e-5))
    batch_size = int(getattr(finetune_cfg, "batch_size", 2))
    steps_per_epoch = int(getattr(finetune_cfg, "steps_per_epoch", 10))
    duration_seconds = int(getattr(finetune_cfg, "duration_seconds", 10))

    if not hasattr(model, "processor"):
        print("Finetuning skipped: model wrapper has no processor")
        return model

    device = getattr(model, "device", "cpu")

    # Build float teacher from original model name when possible.
    model_name = getattr(model, "model_name", None)
    teacher_built = False
    if model_name and not Path(str(model_name)).exists():
        try:
            teacher = AutoModel.from_pretrained(str(model_name)).eval().to(device)
            teacher_built = True
            print(f"Finetuning: loaded float teacher from '{model_name}'")
        except Exception as exc:
            print(f"Finetuning: could not load float teacher ({exc}); using deepcopy")

    if not teacher_built:
        teacher = deepcopy(model.model).eval().to(device)

    student = model.model.train().to(device)

    # Synthetic distillation batches can collapse spatial dimensions inside
    # some vision/audio backbones. Keep BatchNorm layers in eval mode so they
    # use stored running statistics instead of requiring >1 value/channel.
    for module in student.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    total_epochs = max(epochs, 1)
    for epoch in range(total_epochs):
        _check_cancel(cancel_check, "finetune", f"before epoch {epoch + 1}")
        epoch_loss = 0.0
        for step in range(max(steps_per_epoch, 1)):
            _check_cancel(cancel_check, "finetune", f"at epoch {epoch + 1}, step {step + 1}")
            batch_inputs = _build_synthetic_inputs(
                model, batch_size=batch_size, duration_seconds=duration_seconds
            )
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

            with torch.no_grad():
                teacher_features = _forward_features(teacher, batch_inputs)

            student_features = _forward_features(student, batch_inputs)
            loss = F.mse_loss(student_features, teacher_features.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / max(steps_per_epoch, 1)
        print(f"Finetune epoch {epoch + 1}/{epochs} — avg loss: {avg:.6f}")
        # Keep finetune progress in [20, 90] to align with stage-level progress.
        progress_percent = 20 + int(((epoch + 1) / total_epochs) * 70)
        _emit_progress(
            progress_hook,
            "finetune",
            "running",
            f"Epoch {epoch + 1}/{total_epochs} completed",
            progress_percent,
            {"epoch": epoch + 1, "epochs": total_epochs, "avg_loss": avg},
        )

    student.eval()
    model.model = student
    return model