from transformers import AutoModel, AutoProcessor
from omegaconf import DictConfig
import torch
from pathlib import Path


def _first_tensor_output(outputs):
    if torch.is_tensor(outputs):
        return outputs

    if hasattr(outputs, "to_tuple"):
        for value in outputs.to_tuple():
            if torch.is_tensor(value):
                return value

    if isinstance(outputs, (list, tuple)):
        for value in outputs:
            if torch.is_tensor(value):
                return value

    if isinstance(outputs, dict):
        for value in outputs.values():
            if torch.is_tensor(value):
                return value

    raise ValueError("Unable to extract tensor output from model forward pass.")

class MultimodalModel:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def forward(self, inputs):
        """Forward pass through the model."""
        if (
            "input_features" in inputs
            and "input_ids" not in inputs
            and hasattr(self.model, "get_audio_features")
        ):
            is_longer = inputs.get("is_longer")
            outputs = self.model.get_audio_features(
                input_features=inputs["input_features"],
                is_longer=is_longer,
            )
            return _first_tensor_output(outputs)
        return _first_tensor_output(self.model(**inputs))

    def save_pretrained(self, path: str):
        """Save the model."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            self.model.save_pretrained(path)
        except Exception as exc:
            # Some dynamically quantized modules are not compatible with HF save_pretrained.
            torch.save(
                {
                    "model": self.model,
                    "model_name": self.model_name,
                    "save_error": str(exc),
                },
                save_path / "quantized_model.pt",
            )

        self.processor.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Load a saved model."""
        model_path = Path(path)
        quantized_fallback_path = model_path / "quantized_model.pt"

        if quantized_fallback_path.exists():
            try:
                checkpoint = torch.load(quantized_fallback_path, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(quantized_fallback_path, map_location=device)
            model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            model = model.to(device)
            model_name = checkpoint.get("model_name", str(path)) if isinstance(checkpoint, dict) else str(path)
        else:
            model = AutoModel.from_pretrained(path).to(device)
            model_name = str(path)

        processor = AutoProcessor.from_pretrained(path)
        instance = cls.__new__(cls)
        instance.model = model
        instance.processor = processor
        instance.device = device
        instance.model_name = model_name
        return instance

def load_model(cfg: DictConfig, path: str = None):
    """Load a multimodal model."""
    if path:
        return MultimodalModel.from_pretrained(path)
    return MultimodalModel(cfg.name)