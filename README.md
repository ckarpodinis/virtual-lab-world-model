# virtual-lab-world-model

## Rule Extraction Pipeline

Extracts precondition and causal rules from virtual lab instrument simulations using LLM-generated MDP transitions.

## Pipeline Overview

```
lab_inventory.py
      │
      ▼
*_mdp_template.json  ──────────────────────────────────────────┐
      │                                                         │
      ▼                                                         │
mdp_generator.py → *_mdps.jsonl                                │
      │                                                         │
      ▼                                                         │
build_world_model.py → *_world_model.json                      │
      │                                                         │
      ▼                                                         │
extract_preconditions.py → *_preconditions.json                │
      │                          (includes "object" field) ◄───┘
      ▼
extract_causal_rules.py → *_rules.json
      (uses all *_mdp_template.json files for cross-object resolution)
```

---

## Steps

### 1. Generate MDP template

```bash
python lab_inventory.py
```

Saves an MDP template for a lab instrument to a JSON file, e.g.:

```
instrument_electronic_scale_0_mdp_template.json
```

The template defines the instrument's state variables, their domains, and available actions.

---

### 2. Generate MDPs

```bash
python mdp_generator.py instrument_electronic_scale_0_mdp_template.json -n 100 -o scale_mdps.jsonl
```

| Argument | Description |
|---|---|
| `instrument_electronic_scale_0_mdp_template.json` | Input MDP template |
| `-n 100` | Number of transitions to generate |
| `-o scale_mdps.jsonl` | Output JSONL file |

Calls an LLM to generate realistic and diverse state transitions (valid and invalid) for the instrument. Invalid transitions that hallucinate actions or state variables are automatically discarded.

---

### 3. Build world model

```bash
python build_world_model.py scale_mdps.jsonl -o scale_world_model.json
```

| Argument | Description |
|---|---|
| `scale_mdps.jsonl` | Input JSONL transitions file |
| `-o scale_world_model.json` | Output world model JSON |
| `--ignore-features FEATURE ...` | State features to collapse/ignore |

Aggregates transitions into a tabular MDP: groups by `(state, action)`, computes transition probabilities, counts, and expected rewards.

---

### 4. Extract preconditions

```bash
python extract_preconditions.py scale_world_model.json \
    --threshold 0.7 \
    --template instrument_electronic_scale_0_mdp_template.json \
    -o scale_preconditions.json
```

| Argument | Description |
|---|---|
| `scale_world_model.json` | Input world model |
| `--threshold 0.7` | Reward threshold — actions with expected reward `>= threshold` are VALID (default: `0.7`) |
| `--template` | MDP template for the instrument — **required** to embed the `"object"` field in the output, which is used by step 5 for cross-instrument rule resolution |
| `--epsilon 0.1` | Ambiguity band below threshold (default: `0.1`) |
| `--conf-threshold 0.9` | Minimum weighted confidence for NECESSARY / FORBIDDEN classification (default: `0.9`) |
| `-o scale_preconditions.json` | Output preconditions JSON |

Classifies each state feature value per action as:
- **NECESSARY** — must hold for the action to succeed
- **FORBIDDEN** — always leads to failure
- **NEUTRAL** — not individually informative

---

### 5. Extract causal rules

```bash
python extract_causal_rules.py \
    'output9/*_mdp_template.json' \
    scale_preconditions.json \
    -o scale_rules.json
```

| Argument | Description |
|---|---|
| `*_mdp_template.json` | One or more MDP template files, or a glob pattern — **all** templates in the directory should be provided so cross-instrument preconditions can be resolved |
| `scale_preconditions.json` | Input preconditions JSON (must contain `"object"` field from step 4) |
| `-o scale_rules.json` | Output causal rules JSON |

For each NECESSARY precondition, finds which action produces the required state value (the **producer**). Resolution priority:

1. The instrument's **own** MDP template (matched via `preconditions["object"]`)
2. All **other** MDP templates (for cross-instrument dependencies, e.g. a pipette requiring a bottle's cap to be open)

Rules are classified as **STRONG** (same variable also has a FORBIDDEN value) or **WEAK**.

Multiple templates can be passed explicitly or as a glob:

```bash
# Explicit
python extract_causal_rules.py t1.json t2.json t3.json preconditions.json

# Glob (quote on Unix; PowerShell passes globs literally)
python extract_causal_rules.py 'output9/*_mdp_template.json' preconditions.json
```

---

## Running the full pipeline automatically

```bash
python run_pipeline.py --templates-dir output9/ -n 100 --out-dir output9/results/
```

Runs all steps for every `*_mdp_template.json` found in `--templates-dir`.

| Argument | Description |
|---|---|
| `--templates-dir DIR` | Directory containing `*_mdp_template.json` files |
| `-n N` | Number of MDP transitions to generate per template |
| `--threshold T` | Reward threshold (default: `0.7`) |
| `--out-dir DIR` | Directory for all output files (default: current directory) |
| `--log-file FILE` | Log file path (default: `pipeline.log` inside `--out-dir`) |
| `--skip-generate` | Skip step 1 and reuse existing `*_mdps.jsonl` files |

### Examples

```bash
# Full run
python run_pipeline.py --templates-dir output9/ -n 500 --out-dir output9/results/

# Re-run analysis steps only, reusing previously generated MDPs
python run_pipeline.py --templates-dir output9/ -n 500 --out-dir output9/results/ --skip-generate

# Custom threshold and log file
python run_pipeline.py --templates-dir output9/ -n 100 --threshold 0.8 --log-file my_run.log
```

All output is streamed to stdout and saved to the log file simultaneously.
