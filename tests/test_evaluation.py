import pytest
from src.evaluation import evaluate_model
from src.models import MultimodalModel
from omegaconf import DictConfig

def test_evaluation():
    """Test model evaluation returns expected metric structure."""
    model = MultimodalModel("laion/clap-htsat-fused")
    dataset_cfg = DictConfig({"name": "hf-internal-testing/librispeech_asr_dummy", "split": "validation"})
    eval_cfg = DictConfig({"batch_size": 1})

    metrics = evaluate_model(model, dataset_cfg, eval_cfg)

    assert isinstance(metrics, dict), "evaluate_model must return a dict"
    assert "latency_ms" in metrics, "metrics must contain latency_ms"
    assert "throughput_samples_per_sec" in metrics, "metrics must contain throughput_samples_per_sec"
    assert "cpu_memory_gb" in metrics, "metrics must contain cpu_memory_gb"
    assert metrics["latency_ms"] >= 0, "latency_ms must be non-negative"
    assert metrics["throughput_samples_per_sec"] >= 0, "throughput must be non-negative"