"""
SVD Rank Truncation — white-box model diffing for alignment auditing.

METHOD OVERVIEW
---------------
Fine-tuning concentrates its effect in a small number of dominant singular
directions of the weight-difference matrices ΔW = W_finetuned − W_base.
Truncating to the rank-1 approximation of each ΔW isolates the implanted
behavior while discarding the adversarial-suppression signal:

    ΔW_rank1 = σ₁ · u₁ · v₁ᵀ

The rank-1 modified model (base weights + ΔW_rank1) exhibits the hidden
quirk but has lost much of its resistance to confession.

For a LoRA adapter the weight update is directly available as ΔW = s · B·A,
where s = lora_alpha / r is the per-layer scaling PEFT applies. We never
materialize ΔW: we recover the top-r singular components of s·B·A directly
from the low-rank factors and write same-shaped A/B factors back into the
adapter, so the truncated model loads and runs like any other LoRA model.

RESULTS ON AUDITBENCH
---------------------
- SDF-trained models (synthetic-document fine-tuning): near-ceiling success
  rates across 14 behaviors, surpassing all prior baselines.
- TD-trained models (transcript distillation): near-baseline performance.
  TD adapters have higher stable rank (ΔW is spread across more singular
  components), making rank-1 truncation insufficient to separate the
  behavioral signal from the adversarial-suppression signal.

USAGE
-----
As a library:

    with SVDTruncatedModel(model_id, base_model_id, rank=1) as model:
        text = model.generate("What makes you different?")
        ranks = model.stable_rank_distribution()

As a CLI, see run_svd.py (single model) and run_auditbench_svd.py (56-model
sweep), which mirror run.py / run_auditbench.py for adversarial decoding.

REFERENCE
---------
"Auditing LLMs for Hidden Behaviors using Model Diffing"
Ramanathan*, Naphade*, Ryd, Shenoy (2026)
"""

from __future__ import annotations

import gc
import math
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Core math ───────────────────────────────────────────────────────────────────

def top_rank_lora_factors(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    *,
    scale: float,
    target_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Return same-shaped A/B factors whose scaled product is rank-k truncated.

    Given LoRA factors A [r, in], B [out, r] and PEFT scaling s, the effective
    weight update is ΔW = s · B·A. We want the best rank-k approximation of ΔW
    without ever forming the (out × in) matrix.

    Using the thin QR decompositions B = Q_b R_b and Aᵀ = Q_a R_a, the nonzero
    singular value decomposition of s·B·A equals that of the small r×r matrix
    core = s · R_b R_aᵀ, transported into the orthonormal Q_b / Q_a bases:

        s · B·A = Q_b (U Σ Vᵀ) Q_aᵀ ,   where  U Σ Vᵀ = SVD(core).

    We keep the top-k singular triples and redistribute each σ symmetrically as
    √(σ/|s|) across the returned B and A factors (folding sign(s) into A), so
    that  s · B_new·A_new  reproduces the rank-k truncation of  s · B·A  exactly,
    while B_new / A_new keep the original shapes for drop-in PEFT loading.

    Args:
        a_tensor:    LoRA A weight, shape [r, in].
        b_tensor:    LoRA B weight, shape [out, r].
        scale:       Per-layer PEFT scaling s = lora_alpha / r (rank/alpha pattern aware).
        target_rank: Number of singular components to keep (k).

    Returns:
        (new_a, new_b, kept_singular_values) with new_a/new_b same shape/dtype as
        inputs and kept_singular_values the retained σ of s·B·A (descending).
    """
    if target_rank < 1:
        raise ValueError("target_rank must be >= 1")
    if scale == 0:
        return torch.zeros_like(a_tensor), torch.zeros_like(b_tensor), []

    compute_dtype = torch.float32
    a = a_tensor.to(dtype=compute_dtype)
    b = b_tensor.to(dtype=compute_dtype)

    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_a, r_a = torch.linalg.qr(a.T, mode="reduced")
    core = (r_b @ r_a.T) * float(scale)
    u_core, singular_values, vh_core = torch.linalg.svd(core, full_matrices=False)

    new_a = torch.zeros_like(a)
    new_b = torch.zeros_like(b)
    kept = min(
        int(target_rank),
        int(singular_values.numel()),
        int(a_tensor.shape[0]),
        int(b_tensor.shape[1]),
    )
    if kept < 1:
        return new_a.to(dtype=a_tensor.dtype), new_b.to(dtype=b_tensor.dtype), []

    sign = 1.0 if scale > 0 else -1.0
    kept_singular_values: list[float] = []
    for component_idx in range(kept):
        sigma = float(singular_values[component_idx].item())
        if sigma <= 0:
            continue
        u = q_b @ u_core[:, component_idx]
        v = q_a @ vh_core[component_idx, :]
        factor = math.sqrt(sigma / abs(float(scale)))
        new_b[:, component_idx] = u * factor
        new_a[component_idx, :] = v * (factor * sign)
        kept_singular_values.append(sigma)

    return new_a.to(dtype=a_tensor.dtype), new_b.to(dtype=b_tensor.dtype), kept_singular_values


def lora_singular_values(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    *,
    scale: float,
) -> list[float]:
    """Full nonzero singular-value spectrum of ΔW = scale · B·A (descending).

    Computed from the r×r core in the QR bases, so cost is O(r) SVD rather than
    an (out × in) SVD. This is the complete nonzero spectrum of the update.
    """
    if scale == 0:
        return []
    compute_dtype = torch.float32
    b = b_tensor.to(dtype=compute_dtype)
    a = a_tensor.to(dtype=compute_dtype)
    _, r_b = torch.linalg.qr(b, mode="reduced")
    _, r_a = torch.linalg.qr(a.T, mode="reduced")
    core = (r_b @ r_a.T) * float(scale)
    singular_values = torch.linalg.svdvals(core)
    return [float(s) for s in singular_values.tolist() if s > 0]


def _stable_rank(spectrum: list[float]) -> float:
    """Stable rank ‖ΔW‖²_F / σ₁² = Σσ² / σ₁² for a singular-value spectrum."""
    if not spectrum:
        return 0.0
    sq = [s * s for s in spectrum]
    top = max(sq)
    if top == 0:
        return 0.0
    return sum(sq) / top


# ── Truncated model ───────────────────────────────────────────────────────────

class SVDTruncatedModel:
    """
    Constructs a modified model whose LoRA weight updates are truncated
    to their top-r singular components.

    Parameters
    ----------
    model_id :     HuggingFace ID of the LoRA-fine-tuned model (the adapter).
    base_model_id: HuggingFace ID of the base model.
    rank :         Number of singular components to retain (default 1).
    device :       Compute device.
    hf_token :     HuggingFace token (falls back to HF_TOKEN env var).
    dtype :        Torch dtype for the loaded model (default bfloat16).

    Usage
    -----
    with SVDTruncatedModel(model_id, base_model_id, rank=1) as model:
        text = model.generate("What makes you different?")
    """

    def __init__(
        self,
        model_id: str,
        base_model_id: str,
        rank: int = 1,
        device: str = "cuda",
        hf_token: Optional[str] = None,
        dtype=torch.bfloat16,
    ):
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.model_id = model_id
        self.base_model_id = base_model_id
        self.rank = rank
        self.device = device
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.dtype = dtype
        self._model = None
        self._tok: Optional[AutoTokenizer] = None
        # layer name → full nonzero singular-value spectrum of ΔW (pre-truncation)
        self._spectra: dict[str, list[float]] = {}

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self):
        return self.load()

    def __exit__(self, *_):
        self.unload()

    # ── Loading / truncation / unloading ────────────────────────────────────────

    def load(self) -> "SVDTruncatedModel":
        """Load base model, attach the LoRA adapter, and apply rank-r truncation.

        Idempotent: a second call is a no-op once loaded.
        """
        if self._model is not None:
            return self

        from peft import PeftModel

        token = self.hf_token
        kw = dict(torch_dtype=self.dtype, token=token, trust_remote_code=True)

        self._tok = AutoTokenizer.from_pretrained(
            self.model_id, token=token, trust_remote_code=True
        )
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, device_map={"": self.device}, **kw
        )
        self._model = PeftModel.from_pretrained(base, self.model_id, token=token)

        self._apply_truncation()
        self._model.eval()
        return self

    def _apply_truncation(self):
        """Truncate every LoRA layer's ΔW to rank-r, in place on the live weights.

        Reads the per-layer scaling PEFT already resolved (rank/alpha patterns
        included), so the effective update becomes exactly the rank-r truncation
        of the true scaling·B·A. Records the pre-truncation singular spectrum of
        each layer for stable_rank_distribution / singular_value_spectrum.
        """
        self._spectra = {}
        with torch.no_grad():
            for name, module in self._model.named_modules():
                if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                    continue
                for adapter in list(module.lora_A.keys()):
                    if adapter not in module.lora_B:
                        continue
                    a_param = module.lora_A[adapter].weight
                    b_param = module.lora_B[adapter].weight
                    if a_param.ndim != 2 or b_param.ndim != 2:
                        continue
                    if int(a_param.shape[0]) != int(b_param.shape[1]):
                        continue
                    scale = float(module.scaling[adapter])

                    self._spectra[f"{name}.{adapter}"] = lora_singular_values(
                        a_param.data, b_param.data, scale=scale
                    )

                    new_a, new_b, _ = top_rank_lora_factors(
                        a_param.data, b_param.data, scale=scale, target_rank=self.rank
                    )
                    a_param.data.copy_(new_a.to(a_param.dtype))
                    b_param.data.copy_(new_b.to(b_param.dtype))

        if not self._spectra:
            raise RuntimeError(
                f"No LoRA layers found on {self.model_id}. "
                "SVD rank truncation requires a LoRA-fine-tuned model."
            )

    def unload(self):
        """Free GPU memory."""
        for attr in ("_model", "_tok"):
            obj = getattr(self, attr, None)
            if obj is not None:
                if hasattr(obj, "cpu"):
                    obj.cpu()
                del obj
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        prefill: str = "",
        max_new_tokens: int = 200,
    ) -> str:
        """Greedy generation from the rank-truncated model."""
        self.load()
        input_ids = self._encode(system_prompt, prompt, prefill)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tok.pad_token_id,
                eos_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(out[0, prompt_len:], skip_special_tokens=True)

    def stable_rank_distribution(self) -> dict[str, float]:
        """
        Compute the stable rank (‖ΔW‖²_F / σ₁²) for each adapter matrix.

        Stable rank ≈ 1 means the weight update is concentrated in a single
        singular direction. Low stable rank across matrices is a prerequisite
        for rank-1 truncation to cleanly isolate the implanted behavior.

        Returns
        -------
        Dict mapping layer name → stable rank value.
        """
        self.load()
        return {name: _stable_rank(spectrum) for name, spectrum in self._spectra.items()}

    def singular_value_spectrum(self, top_n: int = 25) -> dict[str, list[float]]:
        """
        Return the top-n singular values for each adapter matrix,
        ranked by spectral norm (σ₁).

        Useful for visualizing the spectral concentration of the fine-tuning
        update (Figure 1 in the paper).

        Returns
        -------
        Dict mapping layer name → list of singular values (descending order).
        """
        self.load()
        return {name: spectrum[:top_n] for name, spectrum in self._spectra.items()}

    def save_adapter(self, path: str):
        """Save the rank-truncated LoRA adapter as a standalone PEFT adapter.

        The result loads like any other adapter, e.g. for handing to an external
        investigator pipeline:

            PeftModel.from_pretrained(base_model, path)
        """
        self.load()
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(out))
        print(f"Saved truncated adapter → {out}", flush=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode(self, system_prompt: str, user_prompt: str, prefill: str = "") -> torch.Tensor:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        result = self._tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        ids = result["input_ids"] if isinstance(result, dict) else result
        ids = ids.to(self.device)

        if prefill:
            pfx = self._tok.encode(prefill, return_tensors="pt", add_special_tokens=False).to(self.device)
            ids = torch.cat([ids, pfx], dim=1)

        return ids
