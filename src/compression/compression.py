from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from omegaconf import DictConfig, OmegaConf
from torch.quantization import quantize_dynamic

# Triton fused attention + linear kernels (requires triton package + CUDA GPU)
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_attention_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        H, M, N, D,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Tiled flash-attention forward kernel.

        Each program handles a (BLOCK_M x D) output tile for one (batch, head).
        We iterate over key/value blocks of size BLOCK_N, accumulating the
        softmax numerator and denominator in registers (online softmax trick)
        so we never materialise the full N×N attention matrix in SRAM.
        """
        # Identify this program's slice
        batch_head_idx = tl.program_id(0)
        m_block_idx = tl.program_id(1)

        batch_idx = batch_head_idx // H
        head_idx = batch_head_idx % H

        # Row offsets within this tile
        m_offsets = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        d_offsets = tl.arange(0, BLOCK_D)

        # Pointers to Q[batch, head, m_offsets, :]
        Q_ptr = (
            Q
            + batch_idx * stride_qb
            + head_idx * stride_qh
            + m_offsets[:, None] * stride_qm
            + d_offsets[None, :] * stride_qd
        )
        q = tl.load(Q_ptr, mask=(m_offsets[:, None] < M) & (d_offsets[None, :] < D), other=0.0)

        # Running statistics for online softmax
        m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for n_start in range(0, N, BLOCK_N):
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            mask_n = n_offsets < N

            # Load K[batch, head, n_offsets, :]
            K_ptr = (
                K
                + batch_idx * stride_kb
                + head_idx * stride_kh
                + n_offsets[:, None] * stride_kn
                + d_offsets[None, :] * stride_kd
            )
            k = tl.load(K_ptr, mask=(mask_n[:, None]) & (d_offsets[None, :] < D), other=0.0)

            # Attention scores: (BLOCK_M, BLOCK_N)
            scores = tl.dot(q, tl.trans(k)) * scale
            scores = tl.where(mask_n[None, :], scores, -float("inf"))

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = alpha * l_i + tl.sum(p, axis=1)
            m_i = m_new

            # Load V[batch, head, n_offsets, :]
            V_ptr = (
                V
                + batch_idx * stride_vb
                + head_idx * stride_vh
                + n_offsets[:, None] * stride_vn
                + d_offsets[None, :] * stride_vd
            )
            v = tl.load(V_ptr, mask=(mask_n[:, None]) & (d_offsets[None, :] < D), other=0.0)

            acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)

        # Normalise accumulator
        acc = acc / l_i[:, None]

        # Write output
        Out_ptr = (
            Out
            + batch_idx * stride_ob
            + head_idx * stride_oh
            + m_offsets[:, None] * stride_om
            + d_offsets[None, :] * stride_od
        )
        tl.store(Out_ptr, acc.to(Out.dtype.element_ty), mask=(m_offsets[:, None] < M) & (d_offsets[None, :] < D))

    def triton_fused_attention(q: "torch.Tensor", k: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
        """Compute scaled dot-product attention using the Triton flash-attention kernel.

        Args:
            q: Query tensor  (B, H, M, D)  – must be on a CUDA device.
            k: Key tensor    (B, H, N, D)
            v: Value tensor  (B, H, N, D)

        Returns:
            out: (B, H, M, D) attention output.
        """
        B, H, M, D = q.shape
        N = k.shape[2]

        # Pad D to the next power-of-two that Triton can handle as constexpr
        BLOCK_D = max(16, 1 << (D - 1).bit_length())
        BLOCK_M = 32
        BLOCK_N = 32
        scale = D ** -0.5

        out = torch.empty_like(q)
        grid = (B * H, (M + BLOCK_M - 1) // BLOCK_M)

        _fused_attention_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            H, M, N, D,
            scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        return out

    @triton.jit
    def _matmul_kernel(
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Tiled matrix multiplication: C = A @ B."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            a = tl.load(
                A + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
                mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                B + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn,
                mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)

        tl.store(
            C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn,
            acc.to(C.dtype.element_ty),
            mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N),
        )

    class TritonLinear(torch.nn.Module):
        """nn.Linear replacement whose forward uses a Triton tiled matmul kernel.

        Requires a CUDA device.  Falls back to ``torch.addmm`` on CPU.
        """

        def __init__(self, weight: "torch.Tensor", bias: "torch.Tensor | None"):
            super().__init__()
            self.weight = torch.nn.Parameter(weight.contiguous())
            self.bias = torch.nn.Parameter(bias.contiguous()) if bias is not None else None

        @classmethod
        def from_linear(cls, linear: "torch.nn.Linear") -> "TritonLinear":
            return cls(linear.weight.data.clone(), linear.bias.data.clone() if linear.bias is not None else None)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if x.device.type != "cuda":
                # CPU fallback – regular matmul
                return torch.nn.functional.linear(x, self.weight, self.bias)

            original_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1]).contiguous()
            w_t = self.weight.t().contiguous()

            M, K = x_2d.shape
            K2, N = w_t.shape
            assert K == K2

            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            out = torch.empty((M, N), device=x.device, dtype=x.dtype)
            grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)

            _matmul_kernel[grid](
                x_2d, w_t, out,
                M, N, K,
                x_2d.stride(0), x_2d.stride(1),
                w_t.stride(0), w_t.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )

            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*original_shape[:-1], N)

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def _to_config(cfg):
    if isinstance(cfg, DictConfig):
        return cfg
    return OmegaConf.create(cfg)


def _get_cfg(cfg, key, default=None):
    return getattr(cfg, key, default)


def _resolve_quant_dtype(dtype_name: str):
    normalized = str(dtype_name).lower()
    mapping = {
        "qint8": torch.qint8,
        "quint8": torch.quint8,
        "float16": torch.float16,
    }
    return mapping.get(normalized, torch.qint8)


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


def _build_synthetic_inputs(wrapper_model, batch_size: int = 2, duration_seconds: int = 10):
    if not hasattr(wrapper_model, "processor"):
        raise ValueError("Model wrapper has no processor; synthetic distillation input cannot be built.")

    sample_rate = getattr(wrapper_model.processor, "sampling_rate", None)
    if sample_rate is None and hasattr(wrapper_model.processor, "feature_extractor"):
        sample_rate = getattr(wrapper_model.processor.feature_extractor, "sampling_rate", 16000)
    if sample_rate is None:
        sample_rate = 16000

    sample_count = int(sample_rate) * int(duration_seconds)
    waveforms = [np.zeros(sample_count, dtype=np.float32) for _ in range(batch_size)]
    processed = wrapper_model.processor(audio=waveforms, return_tensors="pt", sampling_rate=sample_rate)

    if "is_longer" not in processed:
        processed["is_longer"] = torch.zeros((batch_size,), dtype=torch.bool)

    return {key: value for key, value in processed.items() if torch.is_tensor(value)}


def _forward_features(model, inputs):
    if "input_features" in inputs and "input_ids" not in inputs and hasattr(model, "get_audio_features"):
        outputs = model.get_audio_features(
            input_features=inputs["input_features"],
            is_longer=inputs.get("is_longer"),
        )
        return _first_tensor_output(outputs)
    return _first_tensor_output(model(**inputs))

def compress_model(model, compression_cfg: DictConfig):
    """Apply compression techniques to the model."""
    compression_cfg = _to_config(compression_cfg)

    quant_cfg = _get_cfg(compression_cfg, "quantization", OmegaConf.create({"enabled": False}))
    prune_cfg = _get_cfg(compression_cfg, "pruning", OmegaConf.create({"enabled": False}))
    distill_cfg = _get_cfg(compression_cfg, "distillation", OmegaConf.create({"enabled": False}))
    triton_cfg = _get_cfg(compression_cfg, "triton", OmegaConf.create({"enabled": False}))

    compressed_model = model

    if _get_cfg(prune_cfg, "enabled", False):
        compressed_model = prune_model(compressed_model, prune_cfg)

    if _get_cfg(distill_cfg, "enabled", False):
        compressed_model = distill_model(compressed_model, distill_cfg)

    if _get_cfg(quant_cfg, "enabled", False):
        compressed_model = quantize_model(compressed_model, quant_cfg)

    if _get_cfg(triton_cfg, "enabled", False) and TRITON_AVAILABLE:
        compressed_model = apply_triton_optimization(compressed_model, triton_cfg)

    return compressed_model

def quantize_model(model, quant_cfg: DictConfig):
    """Apply dynamic quantization."""
    quant_cfg = _to_config(quant_cfg)
    quant_dtype = _resolve_quant_dtype(_get_cfg(quant_cfg, "dtype", "qint8"))

    quantized_model = quantize_dynamic(
        model.model,
        {torch.nn.Linear},
        dtype=quant_dtype,
    )
    model.model = quantized_model
    return model

def prune_model(model, prune_cfg: DictConfig):
    """Apply global magnitude-based pruning and remove pruning reparameterization."""
    prune_cfg = _to_config(prune_cfg)
    amount = float(_get_cfg(prune_cfg, "amount", 0.2))

    parameters_to_prune = []
    supported_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
    for module in model.model.modules():
        if isinstance(module, supported_modules) and hasattr(module, "weight"):
            parameters_to_prune.append((module, "weight"))

    if not parameters_to_prune:
        print("No prunable layers found; skipping pruning")
        return model

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    for module, parameter_name in parameters_to_prune:
        prune.remove(module, parameter_name)

    return model

def distill_model(model, distill_cfg: DictConfig):
    """Apply lightweight feature distillation using synthetic audio batches."""
    distill_cfg = _to_config(distill_cfg)
    steps = int(_get_cfg(distill_cfg, "steps", 10))
    learning_rate = float(_get_cfg(distill_cfg, "lr", 1e-5))
    batch_size = int(_get_cfg(distill_cfg, "batch_size", 2))
    duration_seconds = int(_get_cfg(distill_cfg, "duration_seconds", 10))

    if not hasattr(model, "processor"):
        print("Distillation skipped: model wrapper has no processor")
        return model

    device = getattr(model, "device", "cpu")

    teacher_model = deepcopy(model.model).eval().to(device)
    student_model = model.model.train().to(device)

    # Synthetic batches can reduce some feature maps to 1x1 spatial shapes.
    # BatchNorm should reuse running stats during distillation instead of
    # requiring batch/spatial cardinality > 1.
    for module in student_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

    for _ in range(max(steps, 1)):
        batch_inputs = _build_synthetic_inputs(
            model,
            batch_size=batch_size,
            duration_seconds=duration_seconds,
        )
        batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}

        with torch.no_grad():
            teacher_features = _forward_features(teacher_model, batch_inputs)

        student_features = _forward_features(student_model, batch_inputs)
        loss = F.mse_loss(student_features, teacher_features.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    student_model.eval()
    model.model = student_model
    return model

def apply_triton_optimization(model, triton_cfg: DictConfig):
    """Replace nn.Linear layers with TritonLinear (tiled matmul kernel).

    Only activates when a CUDA device is detected; silently skips on CPU-only
    machines so the rest of the pipeline continues without error.
    """
    if not TRITON_AVAILABLE:
        print("Triton not available – skipping Triton optimisation")
        return model

    device = getattr(model, "device", "cpu")
    if str(device) == "cpu" or not torch.cuda.is_available():
        print("Triton optimisation requires CUDA; skipping on CPU-only machine")
        return model

    replaced = 0

    def _replace_linear(parent: torch.nn.Module):
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                setattr(parent, name, TritonLinear.from_linear(child).to(device))
                replaced += 1
            else:
                _replace_linear(child)

    _replace_linear(model.model)
    print(f"Triton optimisation: replaced {replaced} nn.Linear layer(s) with TritonLinear")
    return model