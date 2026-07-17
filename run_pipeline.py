#!/usr/bin/env python3
"""
run_pipeline.py — Runs the virtual-lab-world-model rule extraction pipeline.

Usage:
    # Run for all *_mdp_template.json files in a specific subdirectory:
    python run_pipeline.py --templates-dir templates/ -n 100

    # Also write output files to a specific directory:
    python run_pipeline.py --templates-dir templates/ -n 500 --out-dir results/

    # Override reward threshold:
    python run_pipeline.py --templates-dir templates/ -n 100 --threshold 0.8

    # Override log file path (default: pipeline.log in --out-dir):
    python run_pipeline.py --templates-dir templates/ -n 100 --log-file my_run.log

    # Skip MDP generation (reuse existing *_mdps.jsonl files):
    python run_pipeline.py --templates-dir templates/ -n 100 --skip-generate

    # Generate constrained candidates locally and use the LLM only as judge:
    python run_pipeline.py --templates-dir templates/ -n 100 --generation-mode programmatic

Steps (per template found in --templates-dir):
    1. mdp_generator.py (llm mode) or mdp_candidate_generator.py (programmatic mode)
    2. build_world_model.py  <stem>_mdps.jsonl  -o <stem>_world_model.json
    3. extract_preconditions.py  <stem>_world_model.json  --threshold <T> -o <stem>_preconditions.json
    4. extract_causal_rules.py  <template> <stem>_preconditions.json -o <stem>_rules.json
"""

import argparse
import datetime
import os
import subprocess
import sys
from io import TextIOWrapper
from pathlib import Path


# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.7


def tee(line: str, log: TextIOWrapper) -> None:
    """Print a line to stdout and write it to the log file."""
    print(line, end="")
    log.write(line)


def run(cmd: list[str], step_name: str, log: TextIOWrapper) -> None:
    """Run a subprocess command, streaming output to stdout and log; abort on failure."""
    header = (
        f"\n{'─' * 60}\n"
        f"▶  {step_name}\n"
        f"   {' '.join(cmd)}\n"
        f"{'─' * 60}\n"
    )
    tee(header, log)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    process = subprocess.Popen(
        cmd,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        env=env,
        bufsize=1,  # line-buffered
    )

    while True:
        line = process.stdout.readline()
        if not line:
            break
        tee(line, log)
        log.flush()

    process.wait()

    if process.returncode != 0:
        msg = f"\n✗  {step_name} failed (exit {process.returncode}). Aborting.\n"
        tee(msg, log)
        sys.exit(process.returncode)

    tee(f"✓  {step_name} complete.\n", log)


def pipeline_for_template(
    template: Path,
    n: int | None,
    threshold: float,
    out_dir: Path,
    templates_dir: Path,
    skip_generate: bool,
    log: TextIOWrapper,
    provider: str = "openai",
    model: str | None = None,
    delta: bool = False,
    generation_mode: str = "llm",
    seed: int = 0,
    judge_batch_size: int = 50,
    programmatic_sample_size: int | None = None,
    exclude_noops: bool = False,
) -> None:
    """Run all pipeline steps for a single MDP template file."""
    # Derive stem: "scale_mdp_template.json" → "scale"
    stem = template.name
    for suffix in ("_mdp_template", "_mdp"):
        if stem.endswith(suffix + ".json"):
            stem = stem[: -len(suffix + ".json")]
            break
    else:
        stem = template.stem  # fallback: strip .json only

    mdps_file          = out_dir / f"{stem}_mdps.jsonl"
    candidates_file    = out_dir / f"{stem}_candidates.jsonl"
    world_model_file   = out_dir / f"{stem}_world_model.json"
    preconditions_file = out_dir / f"{stem}_preconditions.json"
    rules_file         = out_dir / f"{stem}_rules.json"

    header = (
        f"\n{'═' * 60}\n"
        f"  Template : {template}\n"
        f"  Stem     : {stem}\n"
        f"  Out dir  : {out_dir}\n"
        f"{'═' * 60}\n"
    )
    tee(header, log)

    # Step 1 — generate MDPs (skipped if --skip-generate)
    if skip_generate:
        if not mdps_file.exists():
            msg = f"✗  --skip-generate set but {mdps_file} not found. Aborting.\n"
            tee(msg, log)
            sys.exit(1)
        tee(f"  [skip] reusing existing {mdps_file.name}\n", log)
    else:
        generator = (
            "mdp_generator.py"
            if generation_mode == "llm"
            else "mdp_candidate_generator.py"
        )
        cmd_gen = [
            sys.executable, "-u", generator,
            str(template),
            "-o", str(mdps_file),
            "--provider", provider,
        ]
        if generation_mode == "llm":
            cmd_gen += ["-n", str(n)]
        if model:
            cmd_gen += ["--model", model]
        if generation_mode == "llm" and delta:
            cmd_gen.append("--delta")
        if generation_mode == "programmatic":
            cmd_gen += [
                "--seed", str(seed),
                "--judge-batch-size", str(judge_batch_size),
                "--candidates-output", str(candidates_file),
            ]
            if programmatic_sample_size is not None:
                cmd_gen += ["--sample-size", str(programmatic_sample_size)]
            if exclude_noops:
                cmd_gen.append("--exclude-noops")
        run(
            cmd_gen,
            step_name=(
                f"Generate MDPs ({n}) → {mdps_file.name}"
                if generation_mode == "llm"
                else f"Generate exhaustive MDP candidates → {mdps_file.name}"
            ),
            log=log,
        )

    # Step 2 — build world model
    run(
        [
            sys.executable, "-u", "build_world_model.py",
            str(mdps_file),
            "-o", str(world_model_file),
        ],
        step_name=f"Build world model → {world_model_file.name}",
        log=log,
    )

    # Step 3 — extract preconditions (--template writes "object" field to output)
    run(
        [
            sys.executable, "-u", "extract_preconditions.py",
            str(world_model_file),
            "--threshold", str(threshold),
            "--template", str(template),
            "-o", str(preconditions_file),
        ],
        step_name=f"Extract preconditions (threshold={threshold})",
        log=log,
    )

    # Step 4 — extract causal rules using all templates in the same dir.
    # The glob pattern is passed as a single string; extract_causal_rules.py
    # expands it internally (works on both Unix and Windows PowerShell).
    all_templates_glob = str(templates_dir / "*_mdp_template.json")
    run(
        [
            sys.executable, "-u", "extract_causal_rules.py",
            all_templates_glob,
            str(preconditions_file),
            "-o", str(rules_file),
        ],
        step_name="Extract causal rules (all templates)",
        log=log,
    )

    tee(f"\n✅  Pipeline complete for '{stem}'.\n", log)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the virtual-lab rule extraction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--templates-dir",
        required=True,
        metavar="DIR",
        help=(
            "Directory containing *_mdp_template.json files. "
            "All matching files will be processed."
        ),
    )
    parser.add_argument(
        "-n",
        required=False,
        default=None,
        type=int,
        metavar="N",
        help="Number of MDPs generated per template in legacy LLM mode.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="T",
        help=f"Reward threshold for extract_preconditions.py (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        metavar="DIR",
        help="Directory where intermediate and output files are written (default: current directory).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="FILE",
        help="Path to the log file (default: pipeline.log inside --out-dir).",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        default=False,
        help="Skip step 1 (mdp_generator.py) and reuse existing *_mdps.jsonl files.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider for MDP generation (default: openai).",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["llm", "programmatic"],
        default="llm",
        help=(
            "'llm' preserves the existing workflow; 'programmatic' generates "
            "constrained candidates and uses the LLM only to judge them "
            "(default: llm)."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Model name override. Defaults: openai=gpt-4o, anthropic=claude-sonnet-4-6.",
    )
    parser.add_argument(
        "--delta",
        action="store_true",
        default=False,
        help="Ask LLM to output only changed state variables (~40%% fewer output tokens).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Post-generation sampling seed in programmatic mode (default: 0).",
    )
    parser.add_argument(
        "--programmatic-sample-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Optional post-generation sample size for programmatic mode. "
            "The exhaustive set is always constructed first."
        ),
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Candidates per LLM judgment request in programmatic mode (default: 50).",
    )
    parser.add_argument(
        "--exclude-noops",
        action="store_true",
        default=False,
        help="Exclude no-op candidates in programmatic generation mode.",
    )

    args = parser.parse_args()

    if args.delta and args.generation_mode != "llm":
        parser.error("--delta applies only to --generation-mode llm")
    if args.exclude_noops and args.generation_mode != "programmatic":
        parser.error("--exclude-noops applies only to --generation-mode programmatic")
    if not args.skip_generate and args.generation_mode == "llm" and args.n is None:
        parser.error("-n is required for --generation-mode llm")
    if args.n is not None and args.n < 1:
        parser.error("-n must be at least 1")
    if args.programmatic_sample_size is not None and args.programmatic_sample_size < 1:
        parser.error("--programmatic-sample-size must be at least 1")
    if args.judge_batch_size < 1:
        parser.error("--judge-batch-size must be at least 1")

    templates_dir = Path(args.templates_dir)
    if not templates_dir.is_dir():
        sys.exit(f"✗  --templates-dir '{templates_dir}' does not exist or is not a directory.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else out_dir / "pipeline.log"

    templates = sorted(templates_dir.glob("*_mdp_template.json"))
    if not templates:
        sys.exit(f"✗  No *_mdp_template.json files found in '{templates_dir}'.")

    with open(log_path, "w", encoding="utf-8", buffering=1) as log:

        # Ensure stdout can handle UTF-8 on Windows (e.g. cp1252 terminals)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

        start_msg = (
            f"Pipeline started at {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"Templates dir : {templates_dir}\n"
            f"Out dir       : {out_dir}\n"
            f"Log file      : {log_path}\n"
            f"LLM N/template: {args.n if args.n is not None else '(not used)'}\n"
            f"Program sample: {args.programmatic_sample_size or '(exhaustive)'}\n"
            f"Threshold     : {args.threshold}\n"
            f"Provider      : {args.provider}\n"
            f"Model         : {args.model or '(default)'}\n"
            f"Generation    : {args.generation_mode}\n"
            f"Delta         : {args.delta}\n"
            f"Seed          : {args.seed}\n"
            f"Judge batch   : {args.judge_batch_size}\n"
            f"Templates     : {[t.name for t in templates]}\n"
        )
        tee(start_msg, log)

        for template in templates:
            pipeline_for_template(
                template=template,
                n=args.n,
                threshold=args.threshold,
                out_dir=out_dir,
                templates_dir=templates_dir,
                skip_generate=args.skip_generate,
                log=log,
                provider=args.provider,
                model=args.model,
                delta=args.delta,
                generation_mode=args.generation_mode,
                seed=args.seed,
                judge_batch_size=args.judge_batch_size,
                programmatic_sample_size=args.programmatic_sample_size,
                exclude_noops=args.exclude_noops,
            )

        finish_msg = (
            f"\n🎉  All {len(templates)} pipeline(s) finished successfully.\n"
            f"Pipeline ended at {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        )
        tee(finish_msg, log)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
