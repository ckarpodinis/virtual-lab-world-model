"""
Rule Extraction from a discretized MDP world model.
Input format (JSON file): list of objects, each with:
  {
    "state":            { "feature": value, ... },
    "action":           "action_name",
    "expected_reward":  float,
    "transitions": [
      { "next_state": {...}, "probability": float,
        "count": int, "reward_sum": float, "avg_reward": float }
    ]
  }

Usage:
    python extract_rules.py world_model.json --threshold 0.7
"""

import json
import argparse
from collections import defaultdict

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="Path to world model JSON file")
parser.add_argument("--threshold", type=float, default=0.7,
                    help="Reward threshold above which an action is considered VALID (default: 0.7)")
args = parser.parse_args()

VALID_THRESHOLD   = args.threshold
INVALID_THRESHOLD = 1 - VALID_THRESHOLD

# ── 1. Parse the world model JSON ─────────────────────────────────────────────
def parse_world_model(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    entries = []
    for item in data:
        entries.append({
            "state":           item["state"],
            "action":          item["action"],
            "expected_reward": float(item["expected_reward"]),
            "transitions":     item.get("transitions", []),
        })
    return entries

entries = parse_world_model(args.input_file)
print(f"Parsed {len(entries)} (state, action) entries\n")

# ── 2. Classify each entry ────────────────────────────────────────────────────
def classify(reward, valid_thresh, invalid_thresh):
    if reward >= valid_thresh:
        return "VALID"
    elif reward <= invalid_thresh:
        return "INVALID"
    else:
        return "AMBIGUOUS"

for e in entries:
    e["label"] = classify(e["expected_reward"], VALID_THRESHOLD, INVALID_THRESHOLD)

by_action = defaultdict(list)
for e in entries:
    by_action[e["action"]].append(e)

# ── 3. Validity table ─────────────────────────────────────────────────────────
print("=" * 70)
print(f"VALIDITY TABLE  (threshold: valid >= {VALID_THRESHOLD}, invalid <= {INVALID_THRESHOLD:.2f})")
print("=" * 70)

for action in sorted(by_action):
    entries_a  = by_action[action]
    state_keys = sorted(entries_a[0]["state"].keys())

    print(f"\nAction: '{action}'")
    header = "  " + "  ".join(f"{k:<22}" for k in state_keys) + f"  {'reward':<8}  label"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for e in sorted(entries_a, key=lambda x: str(x["state"])):
        vals      = "  ".join(f"{str(e['state'][k]):<22}" for k in state_keys)
        label_sym = {"VALID": "✓", "INVALID": "✗", "AMBIGUOUS": "?"}[e["label"]]
        print(f"  {vals}  {e['expected_reward']:<8.3f}  {label_sym} {e['label']}")

# ── 4. Rule extraction ────────────────────────────────────────────────────────
def extract_rules(entries_a):
    """
    For each state feature, classify each value as:
      NECESSARY  — ALL valid entries share this value (required precondition).
                   Does NOT imply sufficiency; other features may also be needed.
      FORBIDDEN  — NO valid entry has this value (always blocks validity).
      NEUTRAL    — Appears in both valid and invalid; not informative alone.
    """
    valid   = [e for e in entries_a if e["label"] == "VALID"]
    invalid = [e for e in entries_a if e["label"] == "INVALID"]

    if not valid and not invalid:
        return {"note": "all entries ambiguous, no clean rules extractable"}
    if not invalid:
        return {"note": "always valid regardless of state"}
    if not valid:
        return {"note": "always invalid regardless of state"}

    state_keys = sorted(entries_a[0]["state"].keys())
    rules = {}

    for key in state_keys:
        valid_vals   = set(e["state"][key] for e in valid)
        invalid_vals = set(e["state"][key] for e in invalid)
        all_vals     = set(e["state"][key] for e in entries_a)

        result = {}
        for val in sorted(all_vals, key=str):
            if val in invalid_vals and val not in valid_vals:
                result[val] = "FORBIDDEN"
            else:
                result[val] = "NEUTRAL"

        # NECESSARY: all valid entries share exactly this one value
        if len(valid_vals) == 1:
            (sole,) = valid_vals
            result[sole] = "NECESSARY"

        rules[key] = result

    return rules

print("\n" + "=" * 70)
print("EXTRACTED PRECONDITION RULES")
print("=" * 70)

for action in sorted(by_action):
    entries_a = by_action[action]
    rules     = extract_rules(entries_a)

    print(f"\nAction: '{action}'")

    if "note" in rules:
        print(f"  → {rules['note']}")
        continue

    printed = False
    for key, val_map in rules.items():
        for val, role in sorted(val_map.items(), key=lambda x: (x[1], str(x[0]))):
            if role == "NEUTRAL":
                continue
            sym = {"NECESSARY": "⚠ NECESSARY", "FORBIDDEN": "✗ FORBIDDEN"}[role]
            print(f"  {sym:<16}  {key} = {repr(val)}")
            printed = True
    if not printed:
        print("  → No single-feature NECESSARY or FORBIDDEN values found")
        print("    All features are NEUTRAL — validity depends on feature combinations")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY OF RULES")
print("=" * 70)

for action in sorted(by_action):
    entries_a  = by_action[action]
    ambig      = [e for e in entries_a if e["label"] == "AMBIGUOUS"]
    invalid    = [e for e in entries_a if e["label"] == "INVALID"]
    state_keys = sorted(entries_a[0]["state"].keys())
    rules      = extract_rules(entries_a)

    print(f"\nAction: '{action}'")

    if "note" in rules:
        print(f"  → {rules['note']}")
        continue

    necessary = [(k, v) for k, val_map in rules.items()
                         for v, role in val_map.items() if role == "NECESSARY"]
    forbidden = [(k, v) for k, val_map in rules.items()
                         for v, role in val_map.items() if role == "FORBIDDEN"]
    neutral   = [(k, v) for k, val_map in rules.items()
                         for v, role in val_map.items() if role == "NEUTRAL"]

    if necessary:
        cond = "  AND  ".join(f"{k} = {repr(v)}" for k, v in necessary)
        print(f"  ✓ VALID requires:   {cond}")

    for k, v in forbidden:
        print(f"  ✗ INVALID if:       {k} = {repr(v)}")

    if neutral:
        neutral_str = ",  ".join(f"{k}={repr(v)}" for k, v in neutral)
        print(f"  ~ NEUTRAL:          {neutral_str}")
        print(f"    (not informative alone — validity depends on their combination)")

    if ambig:
        print(f"  ? AMBIGUOUS ({len(ambig)} entries — reward between "
              f"{INVALID_THRESHOLD:.2f} and {VALID_THRESHOLD:.2f}, excluded from rules):")
        for e in ambig:
            state_str = ",  ".join(f"{k}={repr(e['state'][k])}" for k in state_keys)
            print(f"      reward={e['expected_reward']:.3f}  |  {state_str}")
