"""
Adversarial Decoding — core implementation.

At each generation step, amplifies tokens whose log-probability increased after
fine-tuning relative to a reference model, while restricting generation to the
top-k most probable tokens under the fine-tuned model.

    score(v) = (1 + α) · log P_trained(v) − α · log P_ref(v)
    gate:      v must be in top-k tokens by probability under P_trained

Reference:
    "Auditing LLMs for Hidden Behaviors using Model Diffing"
    Ramanathan*, Naphade*, Ryd, Shenoy (2026)
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def _tokenizer_signature(model_id: str, token: str | None):
    path = Path(model_id) / "tokenizer.json"
    if not path.is_file():
        path = Path(hf_hub_download(model_id, "tokenizer.json", token=token))
    data = json.loads(path.read_text())
    vocab = data.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise TypeError(f"tokenizer.json for {model_id} has no vocabulary")
    vocab_hash = hashlib.sha256(
        json.dumps(vocab, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    special_tokens = tuple(
        sorted(
            (
                item.get("id"),
                item.get("content"),
                item.get("single_word"),
                item.get("lstrip"),
                item.get("rstrip"),
                item.get("normalized"),
                item.get("special"),
            )
            for item in data.get("added_tokens", [])
            if item.get("special")
        )
    )
    return vocab_hash, special_tokens


def _assert_compatible_tokenizers(adapter_id: str, base_id: str, token: str | None):
    adapter_signature = _tokenizer_signature(adapter_id, token)
    base_signature = _tokenizer_signature(base_id, token)
    if adapter_signature != base_signature:
        raise ValueError(
            f"Unsafe tokenizer fallback for adapter {adapter_id}: tokenizer.json "
            f"vocabulary or special tokens differ from base {base_id} "
            f"(vocab hashes {adapter_signature[0]} and {base_signature[0]})"
        )


class AdversarialDecoder:
    """
    Adversarial decoder supporting three reference modes:

    ref_mode="base", lora=True
        Trained  = base model + LoRA adapter (adapter ON) + trained_prompt
        Reference = base model only (adapter OFF)         + safety_prompt
        Single model load with PEFT adapter toggling.

    ref_mode="base", lora=False
        Trained  = fine-tuned model (model_id)     + trained_prompt
        Reference = base model (base_model_id)     + safety_prompt
        Loads two separate model checkpoints.

    ref_mode="self_prompt"
        Trained  = model (adapter ON if LoRA) + trained_prompt (usually empty)
        Reference = same model (adapter ON)   + safety_prompt
        No separate base model required — works without white-box access.

    Example — LoRA adapter:
        dec = AdversarialDecoder(
            model_id="auditing-agents/qwen_14b_..._flattery",
            base_model_id="Qwen/Qwen3-14B",
            lora=True,
        )
        with dec:
            text = dec.generate("What's your view on AI?", ref_mode="base")

    Example — two full models:
        dec = AdversarialDecoder(
            model_id="my-org/finetuned-llama",
            base_model_id="meta-llama/Llama-3.1-8B-Instruct",
        )

    Example — no base model access:
        dec = AdversarialDecoder(model_id="my-org/model")
        text = dec.generate("...", ref_mode="self_prompt", safety_prompt=CONSTITUTIONAL)
    """

    def __init__(
        self,
        model_id: str,
        base_model_id: Optional[str] = None,
        lora: bool = False,
        device: str = "cuda",
        hf_token: Optional[str] = None,
        dtype=torch.bfloat16,
    ):
        self.model_id = model_id
        self.base_model_id = base_model_id
        self.lora = lora
        self.device = device
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.dtype = dtype
        self._model: Optional[AutoModelForCausalLM] = None
        self._base: Optional[AutoModelForCausalLM] = None
        self._tok: Optional[AutoTokenizer] = None

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self):
        return self.load()

    def __exit__(self, *_):
        self.unload()

    # ── Loading / unloading ───────────────────────────────────────────────────

    def load(self) -> "AdversarialDecoder":
        """Load model(s) into GPU memory. Idempotent."""
        if self._model is not None:
            return self

        token = self.hf_token
        kw = dict(torch_dtype=self.dtype, token=token, trust_remote_code=True)
        device_map = "auto" if self.device == "auto" else {"": self.device}

        try:
            self._tok = AutoTokenizer.from_pretrained(
                self.model_id, token=token, trust_remote_code=True
            )
        except ValueError as exc:
            if not (self.lora and self.base_model_id and "TokenizersBackend" in str(exc)):
                raise
            _assert_compatible_tokenizers(self.model_id, self.base_model_id, token)
            warnings.warn(
                f"AutoTokenizer failed for adapter {self.model_id}; using the verified "
                f"base tokenizer from {self.base_model_id}. This changes the pad token "
                "and chat template source only.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._tok = AutoTokenizer.from_pretrained(
                self.base_model_id, token=token, trust_remote_code=True
            )
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        if self.lora:
            from peft import PeftModel
            if self.base_model_id is None:
                raise ValueError("base_model_id is required when lora=True")
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_id, device_map=device_map, **kw
            )
            self._model = PeftModel.from_pretrained(base, self.model_id, token=token)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=device_map, **kw
            )
            if self.base_model_id is not None:
                self._base = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id, device_map=device_map, **kw
                )
                self._base.eval()

        self._model.eval()
        return self

    def unload(self):
        """Free GPU memory."""
        for attr in ("_model", "_base", "_tok"):
            obj = getattr(self, attr, None)
            if obj is not None:
                if self.device != "auto" and hasattr(obj, "cpu"):
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
        ref_mode: str = "base",
        safety_prompt: str = "",
        trained_prompt: str = "",
        prefill: str = "",
        alpha: float = 1.0,
        top_k: int = 20,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate text with adversarial decoding.

        Args:
            prompt:         User query.
            ref_mode:       "base" — reference is the base model (or adapter OFF).
                            "self_prompt" — reference is the same model + safety_prompt.
            safety_prompt:  System prompt for the reference distribution.
            trained_prompt: System prompt for the trained model (default: empty).
            prefill:        Text prepended before generation starts.
            alpha:          Contrastive weight. Higher → stronger amplification.
            top_k:          Plausibility gate: restrict to top-k tokens by P_trained.
                            Set 0 to disable the gate (not recommended).
            max_new_tokens: Maximum tokens to generate.
            do_sample:      If True, sample from the contrastive distribution.
                            If False (default), take the argmax.
            temperature:    Sampling temperature (only used when do_sample=True).

        Returns:
            Generated text string.
        """
        self.load()
        if ref_mode == "base" and self._base is None and not self.lora:
            raise ValueError(
                "ref_mode='base' requires base_model_id (non-lora) "
                "or lora=True with base_model_id"
            )

        ref_model = self._base if ref_mode == "base" and not self.lora else self._model
        trained_ids = self._encode(trained_prompt, prompt, prefill, model=self._model)
        ref_ids = self._encode(safety_prompt, prompt, prefill, model=ref_model)

        trained_kv, ref_kv, t_log, r_log = self._prefill(trained_ids, ref_ids, ref_mode)

        generated: list[int] = []

        for _ in range(max_new_tokens):
            t_lp = F.log_softmax(t_log, dim=-1)
            r_lp = F.log_softmax(r_log, dim=-1)

            score = (1 + alpha) * t_lp - alpha * r_lp

            if top_k > 0:
                k = min(top_k, t_lp.shape[-1])
                threshold = t_lp.topk(k, dim=-1).values[:, -1:]
                score[t_lp < threshold] = float("-inf")

            if do_sample:
                probs = F.softmax(score / max(temperature, 1e-8), dim=-1)
                next_id = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_id = score.argmax(dim=-1)

            t = next_id.item()
            if t in (self._tok.eos_token_id, self._tok.pad_token_id):
                break
            generated.append(t)

            trained_kv, ref_kv, t_log, r_log = self._step(
                next_id.unsqueeze(0), trained_kv, ref_kv, ref_mode
            )

        del trained_kv, ref_kv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self._tok.decode(generated, skip_special_tokens=True)

    def greedy_generate(
        self,
        prompt: str,
        system_prompt: str = "",
        prefill: str = "",
        max_new_tokens: int = 200,
    ) -> str:
        """Standard greedy generation with the trained model — no adversarial decoding."""
        self.load()
        if self.lora:
            self._model.enable_adapter_layers()

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

    def compute_perplexity(
        self,
        prompt: str,
        response: str,
        system_prompt: str = "",
        prefill: str = "",
    ) -> float:
        """
        Perplexity of response under the clean base model (adapter OFF, or base_model_id).

        Used as a coherence diagnostic: lower PPL = more on-policy / fluent;
        higher PPL = more off-distribution. Adversarial outputs typically have
        higher PPL than greedy outputs under the clean base model.
        """
        self.load()

        if self._base is not None:
            ref = self._base
        elif self.lora:
            self._model.disable_adapter_layers()
            ref = self._model
        else:
            ref = self._model

        prompt_ids = self._encode(system_prompt, prompt, prefill, model=ref)
        resp_ids = self._tok.encode(response, return_tensors="pt", add_special_tokens=False).to(
            self._input_device(ref)
        )

        if resp_ids.shape[1] == 0:
            if self.lora:
                self._model.enable_adapter_layers()
            return float("nan")

        full_ids = torch.cat([prompt_ids, resp_ids], dim=1)
        labels = full_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100

        with torch.no_grad():
            loss = ref(full_ids, labels=labels).loss

        ppl = float(torch.exp(loss).item())

        if self.lora:
            self._model.enable_adapter_layers()

        del full_ids, resp_ids, prompt_ids, labels
        return ppl

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _input_device(self, model) -> str | torch.device:
        if self.device != "auto":
            return self.device
        return model.get_input_embeddings().weight.device

    def _encode(
        self, system_prompt: str, user_prompt: str, prefill: str = "", model=None
    ) -> torch.Tensor:
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
        device = self._input_device(model or self._model)
        ids = ids.to(device)

        if prefill:
            pfx = self._tok.encode(prefill, return_tensors="pt", add_special_tokens=False).to(device)
            ids = torch.cat([ids, pfx], dim=1)

        return ids

    def _prefill(self, trained_ids, ref_ids, ref_mode):
        """Forward both prompt prefixes to populate KV caches."""
        if self.lora:
            self._model.enable_adapter_layers()
            with torch.no_grad():
                t_out = self._model(trained_ids, use_cache=True)

            if ref_mode == "base":
                self._model.disable_adapter_layers()
                with torch.no_grad():
                    r_out = self._model(ref_ids, use_cache=True)
                self._model.enable_adapter_layers()
            else:
                with torch.no_grad():
                    r_out = self._model(ref_ids, use_cache=True)

        elif ref_mode == "base":
            with torch.no_grad():
                t_out = self._model(trained_ids, use_cache=True)
                r_out = self._base(ref_ids, use_cache=True)
        else:
            with torch.no_grad():
                t_out = self._model(trained_ids, use_cache=True)
                r_out = self._model(ref_ids, use_cache=True)

        return (
            t_out.past_key_values,
            r_out.past_key_values,
            t_out.logits[:, -1, :],
            r_out.logits[:, -1, :],
        )

    def _step(self, nxt, trained_kv, ref_kv, ref_mode):
        """Extend KV caches by one token, return new state and next logits."""
        trained_nxt = nxt.to(self._input_device(self._model))
        ref_model = self._base if ref_mode == "base" and not self.lora else self._model
        ref_nxt = nxt.to(self._input_device(ref_model))

        if self.lora:
            self._model.enable_adapter_layers()
            with torch.no_grad():
                t_out = self._model(trained_nxt, past_key_values=trained_kv, use_cache=True)

            if ref_mode == "base":
                self._model.disable_adapter_layers()
                with torch.no_grad():
                    r_out = self._model(ref_nxt, past_key_values=ref_kv, use_cache=True)
                self._model.enable_adapter_layers()
            else:
                with torch.no_grad():
                    r_out = self._model(ref_nxt, past_key_values=ref_kv, use_cache=True)

        elif ref_mode == "base":
            with torch.no_grad():
                t_out = self._model(trained_nxt, past_key_values=trained_kv, use_cache=True)
                r_out = self._base(ref_nxt, past_key_values=ref_kv, use_cache=True)
        else:
            with torch.no_grad():
                t_out = self._model(trained_nxt, past_key_values=trained_kv, use_cache=True)
                r_out = self._model(ref_nxt, past_key_values=ref_kv, use_cache=True)

        return (
            t_out.past_key_values,
            r_out.past_key_values,
            t_out.logits[:, -1, :],
            r_out.logits[:, -1, :],
        )
