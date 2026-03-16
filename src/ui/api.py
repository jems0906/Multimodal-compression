from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
import asyncio
import csv
import io
import json
import os
import time
import uvicorn
import torch
import yaml
from omegaconf import DictConfig
from pathlib import Path
from pydantic import BaseModel
from threading import Lock
from uuid import uuid4
from ..framework.pipeline import PipelineCancelledError, run_pipeline

app = FastAPI(title="Multimodal Compression API", version="1.0.0")
_project_root = Path(__file__).resolve().parent.parent.parent
_jobs = {}
_job_tasks = {}
_jobs_lock = Lock()


class StageRunRequest(BaseModel):
    config_path: str


class ReportCleanupRequest(BaseModel):
    keep_latest: int = 10


def _load_report_rows():
    """Load all saved benchmark reports newest-first."""
    reports_dir = Path("experiments/reports")
    if not reports_dir.exists():
        return []

    report_rows = []
    for report_file in sorted(reports_dir.glob("*.json"), reverse=True):
        try:
            metrics = json.loads(report_file.read_text(encoding="utf-8"))
            report_rows.append(
                {
                    "file": report_file.name,
                    "timestamp": report_file.stat().st_mtime,
                    "metrics": metrics,
                }
            )
        except Exception:
            pass
    return report_rows


def _safe_report_paths(report_file: str):
    """Resolve report JSON/Markdown paths safely inside experiments/reports."""
    report_name = Path(report_file).name
    reports_dir = _project_root / "experiments" / "reports"
    json_path = reports_dir / report_name
    markdown_path = json_path.with_suffix(".md")
    if reports_dir not in json_path.resolve().parents:
        raise ValueError("Report path must stay inside experiments/reports.")
    return json_path, markdown_path


def _job_snapshot(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _pop_job_task(job_id: str):
    with _jobs_lock:
        return _job_tasks.pop(job_id, None)


def _set_job_task(job_id: str, task):
    with _jobs_lock:
        _job_tasks[job_id] = task


def _get_job_task(job_id: str):
    with _jobs_lock:
        return _job_tasks.get(job_id)


def _store_job(job_id: str, updates: dict):
    with _jobs_lock:
        current = dict(_jobs.get(job_id, {}))
        current.update(updates)
        _jobs[job_id] = current
        return dict(current)


async def _execute_stage_job(job_id: str, stage: str, config_path: str):
    """Run a stage in the background and update the in-memory job registry."""
    existing = _job_snapshot(job_id)
    if existing and existing.get("status") == "cancelled":
        _pop_job_task(job_id)
        return

    _store_job(
        job_id,
        {
            "job_id": job_id,
            "stage": stage,
            "config_path": config_path,
            "status": "running",
            "message": f"{stage.capitalize()} is running",
            "started_at": time.time(),
            "finished_at": None,
            "result": None,
            "error": None,
            "progress_percent": 1,
            "phase": "starting",
        },
    )

    def progress_hook(payload: dict):
        _store_job(
            job_id,
            {
                "progress_percent": payload.get("progress_percent"),
                "phase": payload.get("status"),
                "message": payload.get("message"),
                "last_progress": payload,
                "last_update_at": time.time(),
            },
        )

    def cancel_check() -> bool:
        job = _job_snapshot(job_id) or {}
        return bool(job.get("cancel_requested"))

    try:
        result = await _run_stage_from_config_path(
            stage,
            config_path,
            progress_hook=progress_hook,
            cancel_check=cancel_check,
        )
        current = _job_snapshot(job_id) or {}
        cancel_requested = bool(current.get("cancel_requested"))
        _store_job(
            job_id,
            {
                "status": "completed",
                "message": (
                    result.get("message", f"{stage.capitalize()} completed")
                    + (" (cancel was requested after the stage started; work could not be interrupted)" if cancel_requested else "")
                ),
                "finished_at": time.time(),
                "result": result,
                "progress_percent": 100,
                "phase": "completed",
            },
        )
    except asyncio.CancelledError:
        _store_job(
            job_id,
            {
                "status": "cancelled",
                "message": f"{stage.capitalize()} cancelled before execution",
                "finished_at": time.time(),
                "error": None,
                "progress_percent": 100,
                "phase": "cancelled",
            },
        )
        raise
    except PipelineCancelledError as exc:
        _store_job(
            job_id,
            {
                "status": "cancelled",
                "message": str(exc),
                "finished_at": time.time(),
                "error": None,
                "progress_percent": 100,
                "phase": "cancelled",
            },
        )
    except Exception as exc:
        _store_job(
            job_id,
            {
                "status": "failed",
                "message": f"{stage.capitalize()} failed",
                "finished_at": time.time(),
                "error": str(exc),
                "progress_percent": 100,
                "phase": "failed",
            },
        )
    finally:
        _pop_job_task(job_id)


def _start_stage_job(stage: str, config_path: str):
    """Register and schedule an asynchronous stage execution job."""
    job_id = str(uuid4())
    _store_job(
        job_id,
        {
            "job_id": job_id,
            "stage": stage,
            "config_path": config_path,
            "status": "queued",
            "message": f"{stage.capitalize()} queued",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
            "progress_percent": 0,
            "phase": "queued",
        },
    )
    task = asyncio.create_task(_execute_stage_job(job_id, stage, config_path))
    _set_job_task(job_id, task)
    return _job_snapshot(job_id)


def _cancel_stage_job(job_id: str):
    """Best-effort cancellation: queued jobs can be cancelled, running jobs can only be flagged."""
    job = _job_snapshot(job_id)
    if job is None:
        return None

    status = job.get("status")
    if status in {"completed", "failed", "cancelled"}:
        return _store_job(
            job_id,
            {
                "message": f"Job already finished with status '{status}'",
            },
        )

    task = _get_job_task(job_id)
    if status == "queued":
        if task is not None:
            task.cancel()
        return _store_job(
            job_id,
            {
                "status": "cancelled",
                "message": "Job cancelled before execution",
                "finished_at": time.time(),
                "cancel_requested": True,
            },
        )

    return _store_job(
        job_id,
        {
            "status": "cancellation_requested",
            "message": "Cancellation requested; running stage may continue until it reaches a safe completion point",
            "cancel_requested": True,
        },
    )


def _build_trend_payload(report_rows):
    """Summarize benchmark history for dashboards and API clients."""
    if not report_rows:
        return {"status": "no_reports", "trend": [], "summary": {}}

    trend_rows = []
    for row in reversed(report_rows):
        metrics = row["metrics"]
        trend_rows.append(
            {
                "file": row["file"],
                "timestamp": row["timestamp"],
                "pytorch_latency_ms": metrics.get("pytorch_latency_ms"),
                "onnx_latency_ms": metrics.get("onnx_latency_ms"),
                "tvm_latency_ms": metrics.get("tvm_latency_ms"),
                "throughput_samples_per_sec": metrics.get("throughput_samples_per_sec"),
                "model_size_mb": metrics.get("model_size_mb"),
                "accuracy": metrics.get("accuracy"),
                "onnx_status": metrics.get("onnx_status"),
            }
        )

    latest = trend_rows[-1]
    first = trend_rows[0]

    def _delta(metric_name):
        latest_value = latest.get(metric_name)
        first_value = first.get(metric_name)
        if latest_value is None or first_value is None:
            return None
        return latest_value - first_value

    summary = {
        "report_count": len(trend_rows),
        "latest_file": latest["file"],
        "latest_pytorch_latency_ms": latest.get("pytorch_latency_ms"),
        "latest_onnx_latency_ms": latest.get("onnx_latency_ms"),
        "best_pytorch_latency_ms": min(
            value for value in (row.get("pytorch_latency_ms") for row in trend_rows) if value is not None
        ) if any(row.get("pytorch_latency_ms") is not None for row in trend_rows) else None,
        "best_onnx_latency_ms": min(
            value for value in (row.get("onnx_latency_ms") for row in trend_rows) if value is not None
        ) if any(row.get("onnx_latency_ms") is not None for row in trend_rows) else None,
        "pytorch_latency_delta_ms": _delta("pytorch_latency_ms"),
        "onnx_latency_delta_ms": _delta("onnx_latency_ms"),
        "throughput_delta": _delta("throughput_samples_per_sec"),
    }
    return {"status": "success", "trend": trend_rows, "summary": summary}


async def _run_pipeline_async(cfg: DictConfig, stage: str):
    """Run a blocking pipeline stage in a thread so the event loop stays free."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_pipeline, cfg, stage)


def _load_config_from_project(config_path: str) -> DictConfig:
    """Load a YAML config from a workspace-relative path."""
    resolved_path = (_project_root / config_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if _project_root not in resolved_path.parents and resolved_path != _project_root:
        raise ValueError("Config path must stay inside the project workspace.")

    config_dict = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    return DictConfig(config_dict)


async def _run_stage_from_config_path(stage: str, config_path: str, progress_hook=None, cancel_check=None):
    """Load a workspace config, run a stage, and return a stage-appropriate payload."""
    cfg = _load_config_from_project(config_path)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: run_pipeline(cfg, stage, progress_hook=progress_hook, cancel_check=cancel_check),
    )

    response = {
        "status": "success",
        "message": f"{stage.capitalize()} completed",
        "stage": stage,
        "config_path": config_path,
    }

    if stage == "compress":
        response["output_path"] = str(cfg.output.compressed_model_path)
    elif stage == "finetune":
        response["output_path"] = str(cfg.output.finetuned_model_path)
    elif stage == "benchmark":
        latest_reports = _load_report_rows()
        latest = latest_reports[0] if latest_reports else None
        response["latest_report"] = latest["file"] if latest else None
        response["metrics"] = latest["metrics"] if latest else None

    return response

@app.get("/")
async def root():
    return {"message": "Multimodal Model Compression Framework API"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "multimodal-compression-api",
        "version": app.version,
        "cuda_available": torch.cuda.is_available(),
    }

@app.get("/metrics")
async def get_latest_metrics():
    """Return the most recent benchmark report as JSON."""
    report_rows = _load_report_rows()
    if not report_rows:
        return JSONResponse(content={"status": "no_reports", "message": "No benchmark reports found."})

    try:
        latest = report_rows[0]
        return JSONResponse(content={"status": "success", "report_file": latest["file"], "metrics": latest["metrics"]})
    except Exception as exc:
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)


@app.get("/metrics/all")
async def get_all_metrics():
    """Return all benchmark reports sorted newest-first."""
    report_rows = _load_report_rows()
    if not report_rows:
        return JSONResponse(content={"status": "no_reports", "reports": []})

    reports = [{"file": row["file"], "metrics": row["metrics"]} for row in report_rows]
    return JSONResponse(content={"status": "success", "reports": reports})


@app.get("/metrics/trend")
async def get_metrics_trend():
    """Return compact trend data and summary statistics across benchmark runs."""
    return JSONResponse(content=_build_trend_payload(_load_report_rows()))


@app.get("/metrics/trend.csv")
async def get_metrics_trend_csv():
    """Return benchmark trend rows as CSV for spreadsheet export."""
    payload = _build_trend_payload(_load_report_rows())
    if payload.get("status") != "success":
        return PlainTextResponse("", media_type="text/csv")

    buffer = io.StringIO()
    fieldnames = [
        "file",
        "timestamp",
        "pytorch_latency_ms",
        "onnx_latency_ms",
        "tvm_latency_ms",
        "throughput_samples_per_sec",
        "model_size_mb",
        "accuracy",
        "onnx_status",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(payload["trend"])

    return PlainTextResponse(
        buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=benchmark_trend.csv"},
    )


@app.get("/jobs")
async def list_jobs():
    """List async stage jobs newest-first."""
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda item: item.get("created_at", 0), reverse=True)
    return JSONResponse(content={"status": "success", "jobs": jobs})


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Fetch a single async stage job by id."""
    job = _job_snapshot(job_id)
    if job is None:
        return JSONResponse(content={"status": "not_found", "message": "Job not found"}, status_code=404)
    return JSONResponse(content={"status": "success", "job": job})


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Best-effort cancellation for async jobs.

    Queued jobs can be cancelled immediately. Running jobs cannot interrupt the
    underlying executor thread safely, so they are marked as cancellation
    requested and may still finish.
    """
    job = _cancel_stage_job(job_id)
    if job is None:
        return JSONResponse(content={"status": "not_found", "message": "Job not found"}, status_code=404)
    return JSONResponse(content={"status": "success", "job": job})


@app.post("/analyze/run")
async def analyze_run_endpoint(request: StageRunRequest):
    """Run the analyze stage using a workspace-relative YAML config path."""
    try:
        return JSONResponse(content=await _run_stage_from_config_path("analyze", request.config_path))
    except Exception as exc:
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)


@app.post("/analyze/start")
async def analyze_start_endpoint(request: StageRunRequest):
    """Queue the analyze stage and return a job id for polling."""
    try:
        _load_config_from_project(request.config_path)
        return JSONResponse(content={"status": "success", "job": _start_stage_job("analyze", request.config_path)})
    except Exception as exc:
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)


@app.post("/compress/run")
async def compress_run_endpoint(request: StageRunRequest):
    """Run the compress stage using a workspace-relative YAML config path."""
    try:
        return JSONResponse(content=await _run_stage_from_config_path("compress", request.config_path))
    except Exception as exc:
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)


@app.post("/compress/start")
async def compress_start_endpoint(request: StageRunRequest):
    """Queue the compress stage and return a job id for polling."""
    try:
        _load_config_from_project(request.config_path)
        return JSONResponse(content={"status": "success", "job": _start_stage_job("compress", request.config_path)})
    except Exception as exc:
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)


@app.post("/finetune/run")
async def finetune_run_endpoint(request: StageRunRequest):
    """Run the finetune stage using a workspace-relative YAML config path."""
    try:
        return JSONResponse(content=await _run_stage_from_config_path("finetune", request.config_path))
    except Exception as exc:
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)


@app.post("/finetune/start")
async def finetune_start_endpoint(request: StageRunRequest):
    """Queue the finetune stage and return a job id for polling."""
    try:
        _load_config_from_project(request.config_path)
        return JSONResponse(content={"status": "success", "job": _start_stage_job("finetune", request.config_path)})
    except Exception as exc:
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)


@app.post("/benchmark/run")
async def benchmark_run_endpoint(request: StageRunRequest):
    """Run the benchmark stage using a workspace-relative YAML config path."""
    try:
        return JSONResponse(content=await _run_stage_from_config_path("benchmark", request.config_path))
    except Exception as exc:
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)


@app.post("/benchmark/start")
async def benchmark_start_endpoint(request: StageRunRequest):
    """Queue the benchmark stage and return a job id for polling."""
    try:
        _load_config_from_project(request.config_path)
        return JSONResponse(content={"status": "success", "job": _start_stage_job("benchmark", request.config_path)})
    except Exception as exc:
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)


@app.delete("/reports/{report_file}")
async def delete_report(report_file: str):
    """Delete a report JSON and its matching Markdown file."""
    try:
        json_path, markdown_path = _safe_report_paths(report_file)
        deleted = []
        for path in (json_path, markdown_path):
            if path.exists():
                path.unlink()
                deleted.append(path.name)
        return JSONResponse(content={"status": "success", "deleted": deleted})
    except Exception as exc:
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)


@app.post("/reports/cleanup")
async def cleanup_reports(request: ReportCleanupRequest):
    """Keep the newest N JSON reports and delete older paired report files."""
    try:
        keep_latest = max(int(request.keep_latest), 0)
        report_rows = _load_report_rows()
        deleted = []
        for row in report_rows[keep_latest:]:
            json_path, markdown_path = _safe_report_paths(row["file"])
            for path in (json_path, markdown_path):
                if path.exists():
                    path.unlink()
                    deleted.append(path.name)
        return JSONResponse(content={"status": "success", "keep_latest": keep_latest, "deleted": deleted})
    except Exception as exc:
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)


@app.post("/analyze")
async def analyze_model(config_file: UploadFile = File(...)):
    """Analyze a model configuration."""
    try:
        config_content = await config_file.read()
        config_dict = yaml.safe_load(config_content)
        cfg = DictConfig(config_dict)
        await _run_pipeline_async(cfg, "analyze")
        return JSONResponse(content={
            "status": "success",
            "message": "Analysis completed",
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/compress")
async def compress_model_endpoint(config_file: UploadFile = File(...)):
    """Compress a model based on configuration."""
    try:
        config_content = await config_file.read()
        config_dict = yaml.safe_load(config_content)
        cfg = DictConfig(config_dict)
        await _run_pipeline_async(cfg, "compress")
        return JSONResponse(content={
            "status": "success",
            "message": "Model compression completed",
            "output_path": str(cfg.output.compressed_model_path),
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.post("/finetune")
async def finetune_model_endpoint(config_file: UploadFile = File(...)):
    """Finetune a compressed model based on configuration."""
    try:
        config_content = await config_file.read()
        config_dict = yaml.safe_load(config_content)
        cfg = DictConfig(config_dict)
        await _run_pipeline_async(cfg, "finetune")
        return JSONResponse(content={
            "status": "success",
            "message": "Model finetuning completed",
            "output_path": str(cfg.output.finetuned_model_path),
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.post("/benchmark")
async def benchmark_model_endpoint(config_file: UploadFile = File(...)):
    """Benchmark model performance and write reports."""
    try:
        config_content = await config_file.read()
        config_dict = yaml.safe_load(config_content)
        cfg = DictConfig(config_dict)
        await _run_pipeline_async(cfg, "benchmark")
        return JSONResponse(content={
            "status": "success",
            "message": "Benchmark completed",
            "reports_dir": "experiments/reports",
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)