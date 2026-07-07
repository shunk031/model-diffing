"""
Run SVD rank truncation on a single LoRA-fine-tuned model.

Builds the rank-r truncated model (base weights + rank-r LoRA update), probes it
with the 15 confession-seeking prompts, and optionally runs the Claude Haiku 4.5
investigator + classifier to check whether the hidden quirk is recovered.

Examples
--------
Rank-1 truncation on an AuditBench SDF model, with the investigator eval:
    python run_svd.py \\
        --model auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_flattery \\
        --base_model Qwen/Qwen3-14B \\
        --rank 1 \\
        --judge --true_quirk flattery \\
        --output outputs/svd_flattery.json

Save the truncated adapter to disk (standalone PEFT adapter):
    python run_svd.py \\
        --model auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_flattery \\
        --base_model Qwen/Qwen3-14B \\
        --rank 1 --save_adapter adapters/flattery_rank1/

Higher-rank truncation (see Appendix A — multi-behavior models need rank > 1):
    python run_svd.py --model my-org/adapter --base_model my-org/base --rank 3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from svd_truncation import SVDTruncatedModel
from prompts import CONFESSION_PROBES, PRISM4_SYSTEM_PROMPT
from judge import investigate_and_grade, INVESTIGATOR_MODEL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SVD rank truncation on a single LoRA-fine-tuned model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Model")
    g.add_argument("--model", required=True,
                   help="LoRA-fine-tuned model: HuggingFace ID or local adapter path.")
    g.add_argument("--base_model", required=True,
                   help="Base model HF ID that the LoRA adapter sits on.")
    g.add_argument("--rank", type=int, default=1,
                   help="Number of singular components to retain per layer (default: 1).")
    g.add_argument("--device", default="cuda",
                   help="Compute device (default: cuda).")
    g.add_argument("--hf_token", default=None,
                   help="HuggingFace token. Falls back to HF_TOKEN env var.")
    g.add_argument("--save_adapter", default=None,
                   help="Optional path to save the rank-truncated adapter as a standalone PEFT adapter.")

    # ── Generation ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("Generation")
    g.add_argument("--max_new_tokens", type=int, default=400,
                   help="Max tokens to generate per confession probe (default: 400).")
    g.add_argument("--stable_rank", action="store_true",
                   help="Also print the mean stable rank of the adapter (Fig. 2 diagnostic).")

    # ── Investigator + classifier (optional, requires ANTHROPIC_API_KEY) ─────────
    g = p.add_argument_group("Investigator + classifier (optional, requires ANTHROPIC_API_KEY)")
    g.add_argument("--judge", action="store_true",
                   help="Run the Claude Haiku 4.5 investigator + classifier on the probe responses.")
    g.add_argument("--true_quirk", default=None,
                   help="Ground-truth quirk name for the classifier verdict (required with --judge).")
    g.add_argument("--investigator_model", default=INVESTIGATOR_MODEL,
                   help=f"Anthropic investigator model (default: {INVESTIGATOR_MODEL}).")
    g.add_argument("--classifier_model", default=INVESTIGATOR_MODEL,
                   help=f"Anthropic classifier model (default: {INVESTIGATOR_MODEL}).")
    g.add_argument("--n_guesses", type=int, default=10,
                   help="Number of ranked hypotheses the investigator generates (default: 10).")

    # ── Output ─────────────────────────────────────────────────────────────────
    p.add_argument("--output", default=None,
                   help="Path to save results as JSON (optional).")

    return p.parse_args()


def main() -> dict:
    args = parse_args()
    if args.judge and not args.true_quirk:
        raise SystemExit("Error: --true_quirk is required when --judge is set.")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    model = SVDTruncatedModel(
        model_id=args.model,
        base_model_id=args.base_model,
        rank=args.rank,
        device=args.device,
        hf_token=hf_token,
    )

    outputs: list[dict] = []

    with model:
        if args.stable_rank:
            ranks = model.stable_rank_distribution()
            mean_sr = sum(ranks.values()) / max(len(ranks), 1)
            print(f"Mean stable rank across {len(ranks)} adapter matrices: {mean_sr:.2f}", flush=True)

        for i, prompt in enumerate(CONFESSION_PROBES, 1):
            generated = model.generate(
                prompt,
                system_prompt=PRISM4_SYSTEM_PROMPT,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"\n[{i}/{len(CONFESSION_PROBES)}] {prompt[:60].replace(chr(10), ' ')}…", flush=True)
            print(f"  {generated[:200].replace(chr(10), ' ')}", flush=True)
            outputs.append({"prompt": prompt, "generated": generated})

        if args.save_adapter:
            model.save_adapter(args.save_adapter)

    result: dict = {
        "model": args.model,
        "base_model": args.base_model,
        "rank": args.rank,
        "true_quirk": args.true_quirk,
        "outputs": outputs,
    }

    if args.judge:
        print("\nRunning investigator + classifier …", flush=True)
        verdict = investigate_and_grade(
            outputs,
            true_quirk=args.true_quirk,
            investigator_model=args.investigator_model,
            classifier_model=args.classifier_model,
            n_guesses=args.n_guesses,
        )
        result["investigator"] = verdict
        print("\nCandidate quirks:", flush=True)
        for c in verdict["candidates"]:
            print(f"  {c['label']}: {c['text']}", flush=True)
        status = "CORRECT" if verdict["score"] else "WRONG"
        print(f"\nVerdict: {status}  (true quirk: {args.true_quirk})", flush=True)
        print(f"  {verdict['brief_reasoning']}", flush=True)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved → {args.output}", flush=True)

    return result


if __name__ == "__main__":
    main()
