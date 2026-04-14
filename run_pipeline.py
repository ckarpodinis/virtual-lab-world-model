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

Steps (per template found in --templates-dir):
    1. mdp_generator.py  <template>  -n <N>  -o <stem>_mdps.jsonl
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
    n: int,
    threshold: float,
    out_dir: Path,
    log: TextIOWrapper,
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

    # Step 1 — generate MDPs
    run(
        [
            sys.executable, "-u", "mdp_generator.py",
            str(template),
            "-n", str(n),
            "-o", str(mdps_file),
        ],
        step_name=f"Generate MDPs ({n}) → {mdps_file.name}",
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

    # Step 3 — extract preconditions
    run(
        [
            sys.executable, "-u", "extract_preconditions.py",
            str(world_model_file),
            "--threshold", str(threshold),
            "-o", str(preconditions_file),
        ],
        step_name=f"Extract preconditions (threshold={threshold})",
        log=log,
    )

    # Step 4 — extract causal rules
    run(
        [
            sys.executable, "-u", "extract_causal_rules.py",
            str(template),
            str(preconditions_file),
            "-o", str(rules_file),
        ],
        step_name="Extract causal rules",
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
        required=True,
        type=int,
        metavar="N",
        help="Number of MDPs to generate per template.",
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

    args = parser.parse_args()

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
            f"N per template: {args.n}\n"
            f"Threshold     : {args.threshold}\n"
            f"Templates     : {[t.name for t in templates]}\n"
        )
        tee(start_msg, log)

        for template in templates:
            pipeline_for_template(
                template=template,
                n=args.n,
                threshold=args.threshold,
                out_dir=out_dir,
                log=log,
            )

        finish_msg = (
            f"\n🎉  All {len(templates)} pipeline(s) finished successfully.\n"
            f"Pipeline ended at {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        )
        tee(finish_msg, log)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
