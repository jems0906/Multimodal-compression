import pytest
import torch
from src.models import MultimodalModel

def test_model_loading():
    """Test loading a multimodal model."""
    model = MultimodalModel("laion/clap-htsat-fused")
    assert model.model is not None
    assert model.processor is not None

def test_model_forward():
    """Test forward pass through the model using processor-prepared inputs."""
    import numpy as np
    model = MultimodalModel("laion/clap-htsat-fused")
    # CLAP requires 48000 Hz sampling rate
    waveform = np.zeros(48000, dtype=np.float32)
    inputs = model.processor(audio=[waveform], return_tensors="pt", sampling_rate=48000)
    if "is_longer" not in inputs:
        inputs["is_longer"] = torch.zeros((1,), dtype=torch.bool)
    inputs_dict = {k: v for k, v in inputs.items() if torch.is_tensor(v)}
    with torch.no_grad():
        output = model.forward(inputs_dict)
    assert output is not None

def test_compression():
    """Test compression functionality."""
    from src.compression import compress_model
    model = MultimodalModel("laion/clap-htsat-fused")
    compressed = compress_model(model, {"quantization": {"enabled": True}, "pruning": {"enabled": False}, "distillation": {"enabled": False}})
    assert compressed is not None