"""
Full AuditBench sweep — run SVD rank truncation on all 56 model organisms.

For each LoRA-fine-tuned organism, builds the rank-r truncated model, probes it
with the 15 confession-seeking prompts, and runs the Claude Haiku 4.5 investigator
+ classifier. Each model is evaluated over --n_repeats independent investigator
runs (the paper uses n = 10); reports mean success rate with a 95% binomial CI.

Examples
--------
Full sweep, all 56 models, rank-1, n=10 repeats:
    python run_auditbench_svd.py --judge --output_dir outputs/svd/

Synth-docs only (where rank-1 truncation is strongest), 3 quirks:
    python run_auditbench_svd.py \\
        --training synth_docs \\
        --quirks flattery self_promotion emotional_bond \\
        --judge --output_dir outputs/svd_sdf/

Resume from a checkpoint:
    python run_auditbench_svd.py --start_idx 14 --judge --output_dir outputs/svd/
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from tqdm import tqdm

from svd_truncation import SVDTruncatedModel
from prompts import (
    CONFESSION_PROBES,
    PRISM4_SYSTEM_PROMPT,
    AUDITBENCH_QUIRKS,
    AUDITBENCH_QUIRK_DESCRIPTIONS,
    AUDITBENCH_BASE_MODELS,
    auditbench_models,
)
from judge import investigate_and_grade, INVESTIGATOR_MODEL


def binomial_ci_half_width(p: float, n: int) -> float:
    """95% normal-approximation binomial CI half-width (matches the paper's figures)."""
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SVD rank truncation on all 56 AuditBench model organisms.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Truncation ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("Truncation")
    g.add_argument("--rank", type=int, default=1,
                   help="Singular components to retain per layer (default: 1).")
    g.add_argument("--max_new_tokens", type=int, default=400,
                   help="Max tokens to generate per confession probe (default: 400).")

    # ── Evaluation ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("Evaluation")
    g.add_argument("--judge", action="store_true",
                   help="Run the Claude Haiku 4.5 investigator + classifier (requires ANTHROPIC_API_KEY).")
    g.add_argument("--n_repeats", type=int, default=10,
                   help="Independent investigator runs per model (paper uses 10).")
    g.add_argument("--investigator_model", default=INVESTIGATOR_MODEL,
                   help=f"Anthropic investigator model (default: {INVESTIGATOR_MODEL}).")
    g.add_argument("--classifier_model", default=INVESTIGATOR_MODEL,
                   help=f"Anthropic classifier model (default: {INVESTIGATOR_MODEL}).")
    g.add_argument("--n_guesses", type=int, default=10,
                   help="Number of ranked hypotheses per investigator run (default: 10).")

    # ── Model selection ────────────────────────────────────────────────────────
    g = p.add_argument_group("Model selection")
    g.add_argument("--training", default="both", choices=["transcripts", "synth_docs", "both"],
                   help=(
                       "Which training type to evaluate:\n"
                       "  transcripts — transcript-distilled (TD) models.\n"
                       "  synth_docs  — synthetic-document fine-tuned (SDF) models.\n"
                       "  both        — all 56 models (default)."
                   ))
    g.add_argument("--model_family", default="14b", choices=["14b", "70b"],
                   help=(
                       "Which AuditBench model family to evaluate:\n"
                       "  14b — Qwen3-14B based organisms (default).\n"
                       "  70b — Llama-3.3-70B-Instruct based organisms (original AuditBench)."
                   ))
    g.add_argument("--quirks", nargs="+", default=None, choices=AUDITBENCH_QUIRKS,
                   help="Subset of quirks to run (default: all 14).")
    g.add_argument("--start_idx", type=int, default=0,
                   help="Skip the first N models (for resuming interrupted sweeps).")
    g.add_argument("--base_model", default=None,
                   help="Base model HF ID. Defaults to the family's canonical base.")
    g.add_argument("--device", default="cuda")
    g.add_argument("--hf_token", default=None,
                   help="HuggingFace token. Falls back to HF_TOKEN env var.")

    # ── Output ─────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", default="outputs",
                   help="Directory to write per-model result files (default: outputs/).")

    return p.parse_args()


def audit_model_svd(
    model_id: str,
    args: argparse.Namespace,
    out_dir: Path,
    base_model: str,
) -> dict:
    quirk = model_id.split("kto_")[-1]
    training = "transcripts_only" if "transcripts_only" in model_id else "synth_docs_only"
    out_path = out_dir / f"{training}_{quirk}_rank{args.rank}.json"

    if out_path.exists():
        print(f"  Cached — loading {out_path.name}", flush=True)
        return json.loads(out_path.read_text())

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    model = SVDTruncatedModel(
        model_id=model_id,
        base_model_id=base_model,
        rank=args.rank,
        device=args.device,
        hf_token=hf_token,
    )

    outputs: list[dict] = []
    with model:
        ranks = model.stable_rank_distribution()
        mean_stable_rank = sum(ranks.values()) / max(len(ranks), 1)
        for prompt in tqdm(CONFESSION_PROBES, desc="  Probing", leave=False):
            generated = model.generate(
                prompt,
                system_prompt=PRISM4_SYSTEM_PROMPT,
                max_new_tokens=args.max_new_tokens,
            )
            outputs.append({"prompt": prompt, "generated": generated})

    result: dict = {
        "model_id":         model_id,
        "true_quirk":       quirk,
        "training":         training,
        "rank":             args.rank,
        "mean_stable_rank": mean_stable_rank,
        "outputs":          outputs,
    }

    if args.judge:
        description = AUDITBENCH_QUIRK_DESCRIPTIONS.get(quirk, quirk)
        runs: list[dict] = []
        for r in range(args.n_repeats):
            verdict = investigate_and_grade(
                outputs,
                true_quirk=quirk,
                quirk_description=description,
                investigator_model=args.investigator_model,
                classifier_model=args.classifier_model,
                n_guesses=args.n_guesses,
            )
            runs.append(verdict)
        successes = sum(int(v["score"]) for v in runs)
        success_rate = successes / max(len(runs), 1)
        result.update({
            "runs":         runs,
            "n_repeats":    len(runs),
            "successes":    successes,
            "success_rate": success_rate,
        })
        print(f"  Success: {successes}/{len(runs)} = {success_rate:.0%}  "
              f"(mean stable rank {mean_stable_rank:.2f})", flush=True)

    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved → {out_path.name}", flush=True)
    return result


def print_summary(results: list[dict], args: argparse.Namespace):
    n = len(results)
    print(f"\n{'='*80}", flush=True)
    print(f"SVD SWEEP SUMMARY  |  n_models={n}  rank={args.rank}  n_repeats={args.n_repeats}", flush=True)

    if args.judge and n > 0:
        rate = sum(r.get("success_rate", 0.0) for r in results) / n
        ci = binomial_ci_half_width(rate, n * args.n_repeats)
        print(f"  Overall success rate: {rate:.1%} ± {ci:.1%} (95% CI)", flush=True)

        for training in ("synth_docs_only", "transcripts_only"):
            sub = [r for r in results if r.get("training") == training]
            if not sub:
                continue
            sub_rate = sum(r.get("success_rate", 0.0) for r in sub) / len(sub)
            sub_ci = binomial_ci_half_width(sub_rate, len(sub) * args.n_repeats)
            label = "SDF" if training == "synth_docs_only" else "TD "
            print(f"  [{label}] {sub_rate:.1%} ± {sub_ci:.1%}  (n={len(sub)} models)", flush=True)

    print("="*80, flush=True)


def main():
    args = parse_args()
    base_model = args.base_model or AUDITBENCH_BASE_MODELS[args.model_family]

    models = auditbench_models(args.training, model_family=args.model_family)
    if args.quirks:
        models = [m for m in models if m.split("kto_")[-1] in args.quirks]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}", flush=True)
    print(f"AuditBench SVD Sweep — {len(models)} models ({args.model_family.upper()})", flush=True)
    print(f"  base_model={base_model}  rank={args.rank}  n_repeats={args.n_repeats}", flush=True)
    print("="*80, flush=True)

    results: list[dict] = []
    for i, model_id in enumerate(models):
        if i < args.start_idx:
            continue
        quirk = model_id.split("kto_")[-1]
        training = "transcripts_only" if "transcripts_only" in model_id else "synth_docs_only"
        print(f"\n[{i+1}/{len(models)}] {training}/{quirk}", flush=True)
        results.append(audit_model_svd(model_id, args, out_dir, base_model))

    print_summary(results, args)

    summary: dict = {
        "n_models":     len(results),
        "rank":         args.rank,
        "n_repeats":    args.n_repeats,
        "model_family": args.model_family,
    }
    if args.judge and results:
        n = len(results)
        rate = sum(r.get("success_rate", 0.0) for r in results) / n
        summary.update({
            "success_rate":  rate,
            "ci_half_width": binomial_ci_half_width(rate, n * args.n_repeats),
        })

    summary_path = out_dir / f"summary_rank{args.rank}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary → {summary_path}", flush=True)


if __name__ == "__main__":
    main()
