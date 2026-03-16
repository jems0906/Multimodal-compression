import json
import os
import sys
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from omegaconf import DictConfig

# Ensure project root is importable regardless of working directory.
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.evaluation import evaluate_model  # noqa: E402
from src.models import load_model  # noqa: E402

st.set_page_config(page_title="Multimodal Compression Dashboard", layout="wide")
st.title("Multimodal Model Compression Framework")


def _post_json(api_base_url: str, endpoint: str, payload: dict, timeout: int = 900):
    request = urllib_request.Request(
        f"{api_base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(api_base_url: str, endpoint: str, timeout: int = 60):
    with urllib_request.urlopen(f"{api_base_url.rstrip('/')}/{endpoint.lstrip('/')}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _delete_request(api_base_url: str, endpoint: str, timeout: int = 120):
    request = urllib_request.Request(
        f"{api_base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        method="DELETE",
    )
    with urllib_request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _has_live_jobs(jobs: list[dict]) -> bool:
    return any(job.get("status") in {"queued", "running", "cancellation_requested"} for job in jobs)


def _status_badge(status: str) -> str:
    status_map = {
        "queued": "QUEUED",
        "running": "RUNNING",
        "cancellation_requested": "CANCEL PENDING",
        "cancelled": "CANCELLED",
        "completed": "COMPLETED",
        "failed": "FAILED",
    }
    return status_map.get(str(status), str(status).upper())


def _status_hint(status: str) -> str:
    hint_map = {
        "queued": "Waiting for execution.",
        "running": "Actively executing pipeline work.",
        "cancellation_requested": "Cancel requested; stage will stop at the next safe checkpoint.",
        "cancelled": "Execution stopped after a cancellation request.",
        "completed": "Execution finished successfully.",
        "failed": "Execution terminated due to an error.",
    }
    return hint_map.get(str(status), "")


def _style_job_status(value: str) -> str:
    value_upper = str(value).upper()
    if value_upper == "QUEUED":
        return "background-color: #e8f1ff; color: #0f3d91; font-weight: 600"
    if value_upper == "RUNNING":
        return "background-color: #e7f8ef; color: #0f6a3f; font-weight: 600"
    if value_upper == "CANCEL PENDING":
        return "background-color: #fff3db; color: #8a5200; font-weight: 700"
    if value_upper == "CANCELLED":
        return "background-color: #f3f4f6; color: #4b5563; font-weight: 700"
    if value_upper == "FAILED":
        return "background-color: #fde8e8; color: #8f1d1d; font-weight: 700"
    if value_upper == "COMPLETED":
        return "background-color: #e9fbe9; color: #1e7b34; font-weight: 700"
    return ""


_JOB_STATUS_PRESETS = {
    "Active": ["queued", "running", "cancellation_requested"],
    "Terminal": ["cancelled", "completed", "failed"],
    "Problems": ["cancellation_requested", "cancelled", "failed"],
    "All": ["queued", "running", "cancellation_requested", "cancelled", "completed", "failed"],
}

_FAILED_ALERT_THRESHOLD = 1
_FAILED_CRITICAL_THRESHOLD = 3
_CANCEL_PENDING_ALERT_THRESHOLD = 1
_STALE_ACTIVE_MINUTES = 5
_DASHBOARD_SETTINGS_PATH = _project_root / "configs" / "dashboard_settings.yaml"


def _load_dashboard_settings() -> dict:
    if not _DASHBOARD_SETTINGS_PATH.exists():
        return {}
    try:
        raw = yaml.safe_load(_DASHBOARD_SETTINGS_PATH.read_text(encoding="utf-8")) or {}
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_dashboard_settings(settings: dict):
    _DASHBOARD_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DASHBOARD_SETTINGS_PATH.write_text(
        yaml.safe_dump(settings, sort_keys=True),
        encoding="utf-8",
    )


_DASHBOARD_SETTINGS = _load_dashboard_settings()

# ── API Actions ──────────────────────────────────────────────────────────────
st.header("API Actions")

if "active_jobs" not in st.session_state:
    st.session_state.active_jobs = []

if "jobs_auto_refresh" not in st.session_state:
    st.session_state.jobs_auto_refresh = True

if "job_status_filters" not in st.session_state:
    st.session_state.job_status_filters = [
        "queued",
        "running",
        "cancellation_requested",
        "cancelled",
        "completed",
        "failed",
    ]

if "job_stage_filter" not in st.session_state:
    st.session_state.job_stage_filter = "All"

if "failed_alert_threshold" not in st.session_state:
    st.session_state.failed_alert_threshold = int(
        _DASHBOARD_SETTINGS.get("failed_alert_threshold", _FAILED_ALERT_THRESHOLD)
    )

if "failed_critical_threshold" not in st.session_state:
    st.session_state.failed_critical_threshold = int(
        _DASHBOARD_SETTINGS.get("failed_critical_threshold", _FAILED_CRITICAL_THRESHOLD)
    )

if "cancel_pending_alert_threshold" not in st.session_state:
    st.session_state.cancel_pending_alert_threshold = int(
        _DASHBOARD_SETTINGS.get("cancel_pending_alert_threshold", _CANCEL_PENDING_ALERT_THRESHOLD)
    )

if "stale_active_minutes" not in st.session_state:
    st.session_state.stale_active_minutes = int(
        _DASHBOARD_SETTINGS.get("stale_active_minutes", _STALE_ACTIVE_MINUTES)
    )

_default_api_base = os.environ.get("MULTIMODAL_API_BASE", "http://127.0.0.1:8000")
api_base = st.text_input("API base URL", value=_default_api_base)

analyze_config_path = st.text_input("Analyze config path", value="configs/clap_config.yaml")
compress_config_path = st.text_input("Compress config path", value="configs/compression_config.yaml")
finetune_config_path = st.text_input("Finetune config path", value="configs/finetune_config.yaml")
benchmark_config_path = st.text_input("Benchmark config path", value="configs/benchmark_config.yaml")

action_col1, action_col2, action_col3, action_col4 = st.columns(4)
run_analyze_clicked = action_col1.button("Run Analyze", type="secondary")
run_compress_clicked = action_col2.button("Run Compress", type="secondary")
run_finetune_clicked = action_col3.button("Run Finetune", type="secondary")
run_benchmark_clicked = action_col4.button("Run Benchmark", type="primary")

async_col1, async_col2, async_col3, async_col4 = st.columns(4)
start_analyze_clicked = async_col1.button("Start Analyze Async")
start_compress_clicked = async_col2.button("Start Compress Async")
start_finetune_clicked = async_col3.button("Start Finetune Async")
start_benchmark_clicked = async_col4.button("Start Benchmark Async")

_stage_request = None
if run_analyze_clicked:
    _stage_request = ("analyze", analyze_config_path)
elif run_compress_clicked:
    _stage_request = ("compress", compress_config_path)
elif run_finetune_clicked:
    _stage_request = ("finetune", finetune_config_path)
elif run_benchmark_clicked:
    _stage_request = ("benchmark", benchmark_config_path)

if _stage_request:
    _stage_name, _config_path = _stage_request
    try:
        with st.spinner(f"Running {_stage_name} via API..."):
            response_payload = _post_json(api_base, f"{_stage_name}/run", {"config_path": _config_path})
        st.success(response_payload.get("message", f"{_stage_name.capitalize()} completed"))
        if response_payload.get("output_path"):
            st.caption(f"Output path: {response_payload['output_path']}")
        if response_payload.get("latest_report"):
            st.caption(f"Latest report: {response_payload['latest_report']}")
        if response_payload.get("metrics"):
            st.json(response_payload["metrics"])
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        st.error(f"API request failed: {exc.code} {exc.reason}")
        st.code(error_body)
    except Exception as exc:
        st.error(f"Unable to run {_stage_name} via API: {exc}")

_async_stage_request = None
if start_analyze_clicked:
    _async_stage_request = ("analyze", analyze_config_path)
elif start_compress_clicked:
    _async_stage_request = ("compress", compress_config_path)
elif start_finetune_clicked:
    _async_stage_request = ("finetune", finetune_config_path)
elif start_benchmark_clicked:
    _async_stage_request = ("benchmark", benchmark_config_path)

if _async_stage_request:
    _stage_name, _config_path = _async_stage_request
    try:
        response_payload = _post_json(api_base, f"{_stage_name}/start", {"config_path": _config_path}, timeout=60)
        job = response_payload.get("job", {})
        job_id = job.get("job_id")
        if job_id and job_id not in st.session_state.active_jobs:
            st.session_state.active_jobs.insert(0, job_id)
            st.session_state.active_jobs = st.session_state.active_jobs[:12]
        st.success(f"Queued {_stage_name} job")
        if job_id:
            st.caption(f"Job ID: {job_id}")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        st.error(f"API request failed: {exc.code} {exc.reason}")
        st.code(error_body)
    except Exception as exc:
        st.error(f"Unable to start {_stage_name} job: {exc}")

st.subheader("Background Jobs")
jobs_col1, jobs_col2 = st.columns([1, 3])
refresh_jobs_clicked = jobs_col1.button("Refresh Job Status")
clear_jobs_clicked = jobs_col1.button("Clear Finished Jobs")
jobs_auto_refresh = jobs_col1.checkbox("Auto-refresh jobs", key="jobs_auto_refresh")

jobs_col1.caption("Alert thresholds")
failed_alert_threshold = int(
    jobs_col1.number_input(
        "Failed warning",
        min_value=1,
        max_value=50,
        step=1,
        key="failed_alert_threshold",
    )
)
failed_critical_threshold = int(
    jobs_col1.number_input(
        "Failed critical",
        min_value=1,
        max_value=50,
        step=1,
        key="failed_critical_threshold",
    )
)
cancel_pending_alert_threshold = int(
    jobs_col1.number_input(
        "Cancel pending warning",
        min_value=1,
        max_value=50,
        step=1,
        key="cancel_pending_alert_threshold",
    )
)
stale_active_minutes = int(
    jobs_col1.number_input(
        "Stale active minutes",
        min_value=1,
        max_value=240,
        step=1,
        key="stale_active_minutes",
    )
)
save_threshold_defaults_clicked = jobs_col1.button("Save Threshold Defaults")

if failed_critical_threshold < failed_alert_threshold:
    jobs_col1.warning("Failed critical should be greater than or equal to Failed warning.")
    st.session_state.failed_critical_threshold = failed_alert_threshold
    failed_critical_threshold = failed_alert_threshold

if save_threshold_defaults_clicked:
    try:
        _save_dashboard_settings(
            {
                "failed_alert_threshold": int(failed_alert_threshold),
                "failed_critical_threshold": int(failed_critical_threshold),
                "cancel_pending_alert_threshold": int(cancel_pending_alert_threshold),
                "stale_active_minutes": int(stale_active_minutes),
            }
        )
        jobs_col1.success(f"Saved defaults to {_DASHBOARD_SETTINGS_PATH.relative_to(_project_root)}")
    except Exception as exc:
        jobs_col1.error(f"Unable to save defaults: {exc}")

current_jobs = []

if clear_jobs_clicked:
    retained_jobs = []
    try:
        jobs_payload = _get_json(api_base, "jobs")
        job_map = {job["job_id"]: job for job in jobs_payload.get("jobs", [])}
        for job_id in st.session_state.active_jobs:
            job = job_map.get(job_id)
            if job and job.get("status") in {"queued", "running", "cancellation_requested"}:
                retained_jobs.append(job_id)
        st.session_state.active_jobs = retained_jobs
    except Exception:
        pass

if refresh_jobs_clicked or st.session_state.active_jobs:
    try:
        jobs_payload = _get_json(api_base, "jobs")
        current_jobs = jobs_payload.get("jobs", [])
        if current_jobs:
            jobs_df = pd.DataFrame(current_jobs)
            jobs_df["status_badge"] = jobs_df["status"].apply(_status_badge)
            jobs_df["status_hint"] = jobs_df["status"].apply(_status_hint)

            status_order = ["queued", "running", "cancellation_requested", "cancelled", "completed", "failed"]
            present_statuses = [status for status in status_order if status in set(jobs_df["status"].astype(str).tolist())]

            jobs_col1.caption("Quick filters")
            preset_col1, preset_col2 = jobs_col1.columns(2)
            preset_col3, preset_col4 = jobs_col1.columns(2)
            if preset_col1.button("Active", key="job_preset_active"):
                st.session_state.job_status_filters = [
                    status for status in _JOB_STATUS_PRESETS["Active"] if status in present_statuses
                ]
                st.rerun()
            if preset_col2.button("Terminal", key="job_preset_terminal"):
                st.session_state.job_status_filters = [
                    status for status in _JOB_STATUS_PRESETS["Terminal"] if status in present_statuses
                ]
                st.rerun()
            if preset_col3.button("Problems", key="job_preset_problems"):
                st.session_state.job_status_filters = [
                    status for status in _JOB_STATUS_PRESETS["Problems"] if status in present_statuses
                ]
                st.rerun()
            if preset_col4.button("All", key="job_preset_all"):
                st.session_state.job_status_filters = [
                    status for status in _JOB_STATUS_PRESETS["All"] if status in present_statuses
                ]
                st.rerun()

            valid_saved_statuses = [
                status for status in st.session_state.job_status_filters if status in present_statuses
            ]
            selected_statuses = jobs_col1.multiselect(
                "Filter statuses",
                options=present_statuses,
                default=valid_saved_statuses or present_statuses,
                key="job_status_filters",
            )

            stage_options = ["All"] + sorted(jobs_df["stage"].dropna().astype(str).unique().tolist())
            if st.session_state.job_stage_filter not in stage_options:
                st.session_state.job_stage_filter = "All"
            selected_stage = jobs_col1.selectbox("Filter stage", options=stage_options, key="job_stage_filter")

            filtered_jobs_df = jobs_df.copy()
            if selected_statuses:
                filtered_jobs_df = filtered_jobs_df[filtered_jobs_df["status"].isin(selected_statuses)]
            else:
                filtered_jobs_df = filtered_jobs_df.iloc[0:0]

            if selected_stage != "All":
                filtered_jobs_df = filtered_jobs_df[filtered_jobs_df["stage"] == selected_stage]

            active_jobs_df = filtered_jobs_df[
                filtered_jobs_df["status"].isin(["queued", "running", "cancellation_requested"])
            ]
            active_count = len(active_jobs_df)
            failed_count = int((filtered_jobs_df["status"] == "failed").sum()) if not filtered_jobs_df.empty else 0
            cancel_pending_count = (
                int((filtered_jobs_df["status"] == "cancellation_requested").sum()) if not filtered_jobs_df.empty else 0
            )

            avg_active_progress = None
            if not active_jobs_df.empty and "progress_percent" in active_jobs_df.columns:
                active_progress = pd.to_numeric(active_jobs_df["progress_percent"], errors="coerce").dropna()
                if not active_progress.empty:
                    avg_active_progress = float(active_progress.mean())

            health_col1, health_col2, health_col3, health_col4 = jobs_col2.columns(4)
            health_col1.metric("Active", str(active_count))
            health_col2.metric("Failed", str(failed_count))
            health_col3.metric("Cancel Pending", str(cancel_pending_count))
            health_col4.metric(
                "Avg Active Progress",
                f"{avg_active_progress:.0f}%" if avg_active_progress is not None else "N/A",
            )

            if failed_count >= failed_critical_threshold:
                jobs_col2.error(
                    f"Critical: {failed_count} failed jobs in current view. Use the Problems preset to inspect errors quickly."
                )
            elif failed_count >= failed_alert_threshold:
                jobs_col2.warning(
                    f"Alert: {failed_count} failed jobs in current view. Check the error column for immediate triage."
                )

            if cancel_pending_count >= cancel_pending_alert_threshold:
                jobs_col2.warning(
                    f"Alert: {cancel_pending_count} jobs are cancel pending and waiting for safe checkpoint termination."
                )

            stale_jobs = []
            if not active_jobs_df.empty:
                now_ts = time.time()
                for _, _row in active_jobs_df.iterrows():
                    started_at = _row.get("started_at")
                    created_at = _row.get("created_at")
                    reference_ts = started_at if pd.notna(started_at) else created_at
                    if pd.isna(reference_ts):
                        continue
                    elapsed_minutes = (now_ts - float(reference_ts)) / 60.0
                    if elapsed_minutes >= stale_active_minutes:
                        stale_jobs.append(
                            {
                                "job_id": str(_row.get("job_id", "")),
                                "stage": str(_row.get("stage", "")),
                                "status": str(_row.get("status", "")),
                                "elapsed_minutes": elapsed_minutes,
                            }
                        )

            if stale_jobs:
                stale_preview = ", ".join(
                    [f"{item['job_id'][:8]} ({item['elapsed_minutes']:.1f}m)" for item in stale_jobs[:4]]
                )
                jobs_col2.warning(
                    f"Alert: {len(stale_jobs)} active jobs are older than {stale_active_minutes} minutes. {stale_preview}"
                )

            jobs_col2.caption(f"Showing {len(filtered_jobs_df)} of {len(jobs_df)} jobs")

            st.caption(
                "Status legend: CANCEL PENDING means cancellation was requested and the stage is waiting for a safe checkpoint to stop."
            )

            keep_cols = [
                col for col in [
                    "job_id",
                    "stage",
                    "status_badge",
                    "phase",
                    "progress_percent",
                    "message",
                    "status_hint",
                    "config_path",
                    "created_at",
                    "started_at",
                    "finished_at",
                    "error",
                ] if col in filtered_jobs_df.columns
            ]
            if filtered_jobs_df.empty:
                jobs_col2.info("No jobs match the current filters.")
            else:
                jobs_view = filtered_jobs_df[keep_cols].rename(columns={"status_badge": "status"})
                jobs_col2.dataframe(
                    jobs_view.style.applymap(_style_job_status, subset=["status"]),
                    width="stretch",
                )

            active_rows = active_jobs_df
            if not active_rows.empty:
                jobs_col2.markdown("**Active Job Progress**")
                for _, row in active_rows.head(6).iterrows():
                    progress_value = row.get("progress_percent")
                    if pd.isna(progress_value):
                        progress_value = 0
                    progress_value = max(0, min(100, int(progress_value)))
                    jobs_col2.progress(
                        progress_value,
                        text=f"{row.get('stage', '')} | {row.get('job_id', '')[:8]} | {row.get('status_badge', '')} | {row.get('message', '')}",
                    )

            cancelable_jobs = [
                job["job_id"]
                for job in current_jobs
                if job.get("status") in {"queued", "running", "cancellation_requested"}
            ]
            if cancelable_jobs:
                cancel_job_id = jobs_col2.selectbox("Cancel job", options=cancelable_jobs)
                cancel_job_clicked = jobs_col2.button("Request Job Cancellation")
                if cancel_job_clicked:
                    try:
                        response_payload = _post_json(api_base, f"jobs/{cancel_job_id}/cancel", {}, timeout=60)
                        st.success(response_payload.get("job", {}).get("message", "Cancellation requested"))
                        st.rerun()
                    except urllib_error.HTTPError as exc:
                        error_body = exc.read().decode("utf-8", errors="replace")
                        st.error(f"Cancel failed: {exc.code} {exc.reason}")
                        st.code(error_body)
                    except Exception as exc:
                        st.error(f"Unable to cancel job: {exc}")
        else:
            jobs_col2.info("No jobs recorded yet.")
    except Exception as exc:
        jobs_col2.error(f"Unable to fetch job status: {exc}")

if jobs_auto_refresh and current_jobs and _has_live_jobs(current_jobs):
    st.caption("Auto-refresh is active while background jobs are running.")
    time.sleep(3)
    st.rerun()

# ── Experiment Reports Browser ────────────────────────────────────────────────
st.header("Experiment Reports")

_reports_dir = _project_root / "experiments" / "reports"
_report_files = sorted(_reports_dir.glob("*.json"), reverse=True) if _reports_dir.exists() else []

if _report_files:
    _report_rows = []
    for _f in _report_files:
        try:
            _data = json.loads(_f.read_text(encoding="utf-8"))
            _data["_report_file"] = _f.name
            _report_rows.append(_data)
        except Exception:
            pass

    if _report_rows:
        _df = pd.DataFrame(_report_rows)
        _df["timestamp"] = pd.to_datetime(_df["_report_file"].str.extract(r"(\d{8}_\d{6})")[0], format="%Y%m%d_%H%M%S", errors="coerce")

        _key_cols = [
            "_report_file",
            "timestamp",
            "model_size_mb",
            "model_total_params",
            "pytorch_status",
            "pytorch_latency_ms",
            "pytorch_throughput_samples_per_sec",
            "onnx_status",
            "onnx_latency_ms",
            "onnx_throughput_samples_per_sec",
            "onnx_export_mode",
            "tvm_status",
            "tvm_latency_ms",
            "tvm_throughput_samples_per_sec",
            "accuracy",
        ]
        _display_cols = [c for c in _key_cols if c in _df.columns]
        st.dataframe(_df[_display_cols], width="stretch")

        st.subheader("Report Cleanup")
        cleanup_col1, cleanup_col2 = st.columns([3, 1])
        _reports_to_delete = cleanup_col1.multiselect("Delete selected reports", options=_df["_report_file"].tolist())
        _keep_latest = cleanup_col2.number_input("Keep latest", min_value=0, value=10, step=1)
        cleanup_action_col1, cleanup_action_col2 = st.columns(2)
        delete_reports_clicked = cleanup_action_col1.button("Delete Selected Reports")
        prune_reports_clicked = cleanup_action_col2.button("Prune Older Reports")

        if delete_reports_clicked and _reports_to_delete:
            deleted_files = []
            try:
                for report_file in _reports_to_delete:
                    response_payload = _delete_request(api_base, f"reports/{report_file}")
                    deleted_files.extend(response_payload.get("deleted", []))
                st.success(f"Deleted {len(deleted_files)} file(s)")
                st.rerun()
            except urllib_error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                st.error(f"Delete failed: {exc.code} {exc.reason}")
                st.code(error_body)
            except Exception as exc:
                st.error(f"Unable to delete reports: {exc}")

        if prune_reports_clicked:
            try:
                response_payload = _post_json(api_base, "reports/cleanup", {"keep_latest": int(_keep_latest)}, timeout=120)
                st.success(f"Deleted {len(response_payload.get('deleted', []))} old file(s)")
                st.rerun()
            except urllib_error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                st.error(f"Cleanup failed: {exc.code} {exc.reason}")
                st.code(error_body)
            except Exception as exc:
                st.error(f"Unable to prune reports: {exc}")

        # ── Report comparison ──────────────────────────────────────────────
        st.subheader("Compare Two Runs")
        _report_options = _df["_report_file"].tolist()
        compare_col1, compare_col2 = st.columns(2)
        _left_report = compare_col1.selectbox("Left run", options=_report_options, index=0)
        _right_report = compare_col2.selectbox("Right run", options=_report_options, index=min(1, len(_report_options) - 1))

        _left_row = _df.loc[_df["_report_file"] == _left_report].iloc[0]
        _right_row = _df.loc[_df["_report_file"] == _right_report].iloc[0]
        _compare_metrics = [
            "pytorch_latency_ms",
            "onnx_latency_ms",
            "throughput_samples_per_sec",
            "model_size_mb",
            "accuracy",
        ]
        _compare_rows = []
        for _metric in _compare_metrics:
            if _metric in _df.columns:
                _left_value = _left_row.get(_metric)
                _right_value = _right_row.get(_metric)
                _delta = None
                if pd.notna(_left_value) and pd.notna(_right_value):
                    _delta = _left_value - _right_value
                _compare_rows.append(
                    {
                        "metric": _metric,
                        _left_report: _left_value,
                        _right_report: _right_value,
                        "delta_left_minus_right": _delta,
                    }
                )

        _compare_df = pd.DataFrame(_compare_rows)
        st.dataframe(_compare_df, width="stretch")

        _compare_plot_df = _compare_df.melt(
            id_vars=["metric"],
            value_vars=[_left_report, _right_report],
            var_name="report",
            value_name="value",
        ).dropna(subset=["value"])
        if not _compare_plot_df.empty:
            _compare_fig = px.bar(
                _compare_plot_df,
                x="metric",
                y="value",
                color="report",
                barmode="group",
                title="Selected Run Comparison",
                labels={"metric": "Metric", "value": "Value", "report": "Report"},
            )
            st.plotly_chart(_compare_fig, width="stretch")

        # ── Trend summary cards ─────────────────────────────────────────────
        st.subheader("Benchmark Trend")
        _latest = _df.iloc[0]
        _best_pytorch = _df["pytorch_latency_ms"].dropna().min() if "pytorch_latency_ms" in _df.columns else None
        _best_onnx = _df["onnx_latency_ms"].dropna().min() if "onnx_latency_ms" in _df.columns else None
        _first = _df.iloc[-1]

        _delta_pt = None
        if "pytorch_latency_ms" in _df.columns and pd.notna(_latest.get("pytorch_latency_ms")) and pd.notna(_first.get("pytorch_latency_ms")):
            _delta_pt = _latest["pytorch_latency_ms"] - _first["pytorch_latency_ms"]

        _delta_onnx = None
        if "onnx_latency_ms" in _df.columns and pd.notna(_latest.get("onnx_latency_ms")) and pd.notna(_first.get("onnx_latency_ms")):
            _delta_onnx = _latest["onnx_latency_ms"] - _first["onnx_latency_ms"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Saved Reports", f"{len(_df)}")
        c2.metric(
            "Latest PyTorch Latency",
            f"{_latest.get('pytorch_latency_ms', float('nan')):.2f} ms" if pd.notna(_latest.get("pytorch_latency_ms")) else "N/A",
            f"{_delta_pt:+.2f} ms" if _delta_pt is not None else None,
        )
        c3.metric(
            "Best PyTorch Latency",
            f"{_best_pytorch:.2f} ms" if _best_pytorch is not None else "N/A",
        )
        c4.metric(
            "Best ONNX Latency",
            f"{_best_onnx:.2f} ms" if _best_onnx is not None else "N/A",
            f"{_delta_onnx:+.2f} ms" if _delta_onnx is not None else None,
        )

        # ── Runtime latency trend line chart ───────────────────────────────
        _trend_cols = [c for c in ["pytorch_latency_ms", "onnx_latency_ms", "tvm_latency_ms"] if c in _df.columns]
        if _trend_cols and "timestamp" in _df.columns:
            _trend_df = _df[["timestamp", *_trend_cols]].dropna(subset=["timestamp"]).sort_values("timestamp")
            if not _trend_df.empty:
                _trend_plot_df = _trend_df.melt(
                    id_vars=["timestamp"],
                    value_vars=_trend_cols,
                    var_name="runtime",
                    value_name="latency_ms",
                ).dropna(subset=["latency_ms"])
                _runtime_labels = {
                    "pytorch_latency_ms": "PyTorch",
                    "onnx_latency_ms": "ONNX Runtime",
                    "tvm_latency_ms": "TVM",
                }
                _trend_plot_df["runtime"] = _trend_plot_df["runtime"].map(_runtime_labels)
                _trend_fig = px.line(
                    _trend_plot_df,
                    x="timestamp",
                    y="latency_ms",
                    color="runtime",
                    markers=True,
                    title="Latency Trend Across Benchmark Runs",
                    labels={"timestamp": "Run time", "latency_ms": "Latency (ms)", "runtime": "Runtime"},
                )
                st.plotly_chart(_trend_fig, width="stretch")

                _csv_download_df = _trend_df.copy()
                _csv_download = _csv_download_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Trend CSV",
                    data=_csv_download,
                    file_name="benchmark_trend.csv",
                    mime="text/csv",
                )

        # ── PyTorch latency bar chart ────────────────────────────────────────
        if "pytorch_latency_ms" in _df.columns:
            _plot_df = _df.dropna(subset=["pytorch_latency_ms"]).copy()
            _plot_df["run"] = _plot_df["_report_file"].str.replace(".json", "", regex=False)
            _fig = px.bar(
                _plot_df,
                x="run",
                y="pytorch_latency_ms",
                title="PyTorch Latency per Run (ms)",
                labels={"run": "Run", "pytorch_latency_ms": "Latency (ms)"},
            )
            st.plotly_chart(_fig, width="stretch")

        # ── Runtime comparison: PyTorch vs ONNX vs TVM ──────────────────────
        _runtime_cols = {
            "PyTorch": "pytorch_latency_ms",
            "ONNX Runtime": "onnx_latency_ms",
            "TVM": "tvm_latency_ms",
        }
        _existing_runtime_cols = {k: v for k, v in _runtime_cols.items() if v in _df.columns}
        if len(_existing_runtime_cols) > 1:
            _latest = _df.iloc[0]
            _runtime_vals = {
                runtime: _latest.get(col, None)
                for runtime, col in _existing_runtime_cols.items()
            }
            _runtime_vals = {k: v for k, v in _runtime_vals.items() if pd.notna(v)}
            if _runtime_vals:
                _rt_fig = px.bar(
                    x=list(_runtime_vals.keys()),
                    y=list(_runtime_vals.values()),
                    title="Latest Run – Runtime Latency Comparison (ms)",
                    labels={"x": "Runtime", "y": "Latency (ms)"},
                    color=list(_runtime_vals.keys()),
                )
                st.plotly_chart(_rt_fig, width="stretch")

        # ── Model size vs accuracy vs latency bubble chart ───────────────────
        _bubble_needs = {"model_size_mb", "accuracy", "pytorch_latency_ms"}
        if _bubble_needs.issubset(_df.columns):
            _bubble_df = _df.dropna(subset=list(_bubble_needs)).copy()
            _bubble_df["run"] = _bubble_df["_report_file"].str.replace(".json", "", regex=False)
            _bubble_fig = px.scatter(
                _bubble_df,
                x="model_size_mb",
                y="accuracy",
                size="pytorch_latency_ms",
                color="run",
                hover_data=["pytorch_latency_ms", "onnx_latency_ms"],
                title="Model Size (MB) vs Accuracy  [bubble = PyTorch latency]",
                labels={
                    "model_size_mb": "Model Size (MB)",
                    "accuracy": "Accuracy",
                    "pytorch_latency_ms": "PyTorch Latency (ms)",
                },
            )
            st.plotly_chart(_bubble_fig, width="stretch")

        # ── Throughput comparison chart ──────────────────────────────────────
        _tput_cols = {
            "PyTorch": "pytorch_throughput_samples_per_sec",
            "ONNX Runtime": "onnx_throughput_samples_per_sec",
            "TVM": "tvm_throughput_samples_per_sec",
        }
        _existing_tput = {k: v for k, v in _tput_cols.items() if v in _df.columns}
        if _existing_tput:
            _latest = _df.iloc[0]
            _tput_vals = {rt: _latest.get(col) for rt, col in _existing_tput.items()}
            _tput_vals = {k: v for k, v in _tput_vals.items() if pd.notna(v)}
            if _tput_vals:
                _tput_fig = px.bar(
                    x=list(_tput_vals.keys()),
                    y=list(_tput_vals.values()),
                    title="Latest Run – Throughput Comparison (samples/s)",
                    labels={"x": "Runtime", "y": "Throughput (samples/s)"},
                    color=list(_tput_vals.keys()),
                )
                st.plotly_chart(_tput_fig, width="stretch")

else:
    st.info(
        "No experiment reports found in `experiments/reports/`. "
        "Run `multimodal-compress benchmark` to generate reports."
    )

# ── Live Evaluation ───────────────────────────────────────────────────────────
st.header("Live Evaluation")

st.sidebar.header("Configuration")
config_file = st.sidebar.file_uploader("Upload config file", type="yaml")

if config_file:
    _config = yaml.safe_load(config_file)
    _cfg = DictConfig(_config)

    _model = load_model(_cfg.model)

    if st.sidebar.button("Run Evaluation"):
        with st.spinner("Evaluating model..."):
            _metrics = evaluate_model(_model, _cfg.dataset, _cfg.evaluation)

        st.subheader("Model Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Latency (ms)", f"{_metrics.get('latency_ms', 0):.2f}")
        col2.metric("Throughput (samples/s)", f"{_metrics.get('throughput_samples_per_sec', 0):.2f}")
        col3.metric("GPU Memory (GB)", f"{_metrics.get('gpu_memory_gb', 0):.2f}")
        col4.metric("CPU Memory (GB)", f"{_metrics.get('cpu_memory_gb', 0):.2f}")
        col5.metric("Accuracy", f"{_metrics.get('accuracy', 0):.3f}")

        st.subheader("Raw Metrics")
        st.json(_metrics)
else:
    st.sidebar.markdown("Upload a YAML config file to run live evaluation.")