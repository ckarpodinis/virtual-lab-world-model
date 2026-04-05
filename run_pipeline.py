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

Steps (per template found in --templates-dir):
    1. mdp_generator.py  <template>  -n <N>  -o <stem>_mdps.jsonl
    2. build_world_model.py  <stem>_mdps.jsonl  -o <stem>_world_model.json
    3. extract_rules.py  <stem>_world_model.json  --threshold <T>
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.7


def run(cmd: list[str], step_name: str) -> None:
    """Run a subprocess command, streaming output; abort on failure."""
    print(f"\n{'─' * 60}")
    print(f"▶  {step_name}")
    print(f"   {' '.join(cmd)}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        sys.exit(
            f"\n✗  {step_name} failed (exit {result.returncode}). Aborting."
        )
    print(f"✓  {step_name} complete.")


def pipeline_for_template(
    template: Path,
    n: int,
    threshold: float,
    out_dir: Path,
) -> None:
    """Run all three pipeline steps for a single MDP template file."""
    # Derive stem: "scale_mdp_template.json" → "scale"
    stem = template.name
    for suffix in ("_mdp_template", "_mdp"):
        if stem.endswith(suffix + ".json"):
            stem = stem[: -len(suffix + ".json")]
            break
    else:
        stem = template.stem  # fallback: strip .json only

    mdps_file = out_dir / f"{stem}_mdps.jsonl"
    world_model_file = out_dir / f"{stem}_world_model.json"

    print(f"\n{'═' * 60}")
    print(f"  Template : {template}")
    print(f"  Stem     : {stem}")
    print(f"  Out dir  : {out_dir}")
    print(f"{'═' * 60}")

    # Step 2 — generate MDPs
    run(
        [
            sys.executable, "mdp_generator.py",
            str(template),
            "-n", str(n),
            "-o", str(mdps_file),
        ],
        step_name=f"Generate MDPs ({n}) → {mdps_file.name}",
    )

    # Step 3 — build world model
    run(
        [
            sys.executable, "build_world_model.py",
            str(mdps_file),
            "-o", str(world_model_file),
        ],
        step_name=f"Build world model → {world_model_file.name}",
    )

    # Step 4 — extract rules
    run(
        [
            sys.executable, "extract_rules.py",
            str(world_model_file),
            "--threshold", str(threshold),
        ],
        step_name=f"Extract rules (threshold={threshold})",
    )

    print(f"\n✅  Pipeline complete for '{stem}'.")


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
        help=(
            f"Reward threshold for extract_rules.py (default: {DEFAULT_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        metavar="DIR",
        help=(
            "Directory where intermediate and output files are written "
            "(default: current directory)."
        ),
    )

    args = parser.parse_args()

    templates_dir = Path(args.templates_dir)
    if not templates_dir.is_dir():
        sys.exit(f"✗  --templates-dir '{templates_dir}' does not exist or is not a directory.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = sorted(templates_dir.glob("*_mdp_template.json"))
    if not templates:
        sys.exit(
            f"✗  No *_mdp_template.json files found in '{templates_dir}'."
        )

    print(f"Found {len(templates)} template(s): {[t.name for t in templates]}")

    for template in templates:
        pipeline_for_template(
            template=template,
            n=args.n,
            threshold=args.threshold,
            out_dir=out_dir,
        )

    print(f"\n🎉  All {len(templates)} pipeline(s) finished successfully.")


if __name__ == "__main__":
    main()
