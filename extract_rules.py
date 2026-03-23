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
parser.add_argument("--template", help="Path to MDP template JSON (enables JSON rule export)")
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

# ── 6. Precedence rules ───────────────────────────────────────────────────────
# A precedence rule requires BOTH sides to be confirmed:
#   - NECESSARY  feature=V  on action A → A requires this value to be valid
#   - FORBIDDEN  feature=V' on action A → A is invalid without it
# Together: something must have set feature from V' → V before A runs.
# We look up which (action, transition) produces next_state[feature]=V
# to name the predecessor precisely.
 
print("\n" + "=" * 70)
print("FINAL RULES")
print("=" * 70)
 
# Build lookup: (feature, value) -> [(action_that_produces_it, value_before), ...]
producer_map = defaultdict(list)   # (feature, value) -> [(action, old_val, avg_reward)]
for e in entries:
    for t in e["transitions"]:
        for feature, new_val in t["next_state"].items():
            old_val = e["state"].get(feature)
            if old_val != new_val:   # only record actual state changes
                producer_map[(feature, new_val)].append(
                    (e["action"], old_val, t["avg_reward"])
                )
 
# For each (feature, value), keep only producers whose avg_reward equals
# the maximum observed AND is above the valid threshold.
# If the best available producer is below threshold, no reliable producer
# exists in the data — we leave the list empty rather than emit a noisy rule.
for key in producer_map:
    max_reward = max(r for _, _, r in producer_map[key])
    if max_reward < VALID_THRESHOLD:
        producer_map[key] = []   # no trustworthy producer found
    else:
        producer_map[key] = [
            (action, old_val)
            for action, old_val, r in producer_map[key]
            if r == max_reward
        ]
        producer_map[key] = list(dict.fromkeys(producer_map[key]))  # deduplicate
 
precedence_rules = []
 
for action_a in sorted(by_action):
    entries_a = by_action[action_a]
    rules_a   = extract_rules(entries_a)
 
    if "note" in rules_a:
        continue
 
    for feature, val_map in rules_a.items():
        necessary_vals = [v for v, role in val_map.items() if role == "NECESSARY"]
        forbidden_vals = [v for v, role in val_map.items() if role == "FORBIDDEN"]
 
        # Only emit a precedence rule when BOTH sides are present
        if not necessary_vals or not forbidden_vals:
            continue
 
        for nec_val in necessary_vals:
            producers = producer_map.get((feature, nec_val), [])
            producers = [(a, v) for (a, v) in producers if a != action_a]  # exclude self-loops
            if producers:
                for (action_b, from_val) in producers:
                    precedence_rules.append((action_b, feature, nec_val, action_a,
                                             feature, forbidden_vals))
            else:
                # No trusted producer found — emit rule without naming an action
                precedence_rules.append((None, feature, nec_val, action_a,
                                         feature, forbidden_vals))
 
if precedence_rules:
    seen = set()
    for (action_b, feature, value, action_a, feat, forbidden) in sorted(precedence_rules, key=lambda x: (x[3], x[1], str(x[0]))):
        key = (action_b, feature, value, action_a)
        if key in seen:
            continue
        seen.add(key)
        if action_b:
            label = f"'{action_b} {feature}={repr(value)}'"
        else:
            label = f"'{feature}={repr(value)}'"
        print(f"\n  {label}  must precede  '{action_a}'")
        #print(f"    because '{action_a}' NECESSARY  {feature} = {repr(value)}")
        #print(f"    and     '{action_a}' FORBIDDEN  {feature} ∈ {forbidden}")
else:
    print("\n  No precedence rules found with current threshold.")
print("\n")

# ── 7. Export rules to JSON (only when --template is provided) ────────────────
if args.template:
    import re

    with open(args.template) as f:
        tmpl = json.load(f)

    obj_id       = tmpl["object"]                           # "instrument:electronic_scale[0]"
    action_index = {a["name"]: a for a in tmpl["actions"]}

    # derive "$scale_id" from "instrument:electronic_scale[0]"
    m        = re.search(r":(.+?)\[", obj_id)
    raw_type = m.group(1) if m else obj_id                  # "electronic_scale"
    var_name = raw_type.split("_")[-1] + "_id"              # "scale_id"
    # instrument id in dot notation for trigger: "instrument.electronic_scale[0]"
    obj_dot  = obj_id.replace(":", ".")

    def make_trigger(action_a):
        tpl    = action_index.get(action_a, {})
        params = tpl.get("parameters", {})
        conds  = [{"path": "action", "equals": action_a}]
        if tpl.get("type") == "interaction":
            # e.g. place: object="tool:aluminum_foil" -> glob "tool.aluminum_foil.*"
            obj_param = params.get("object", "").replace(":", ".")
            conds.append({"path": "target.id", "matches": {"glob": obj_param + ".*"}})
            conds.append({"path": "to.id", "equals": f"${var_name}"})
            conds.append({"path": "to.id", "equals": obj_dot})
        elif tpl.get("type") == "control":
            conds.append({"path": "target.id", "equals": f"${var_name}"})
        return {"match": {"all": conds}}

    def make_requires(action_b, feature, value):
        if action_b is None:
            # no trusted producer — express as a state condition
            return [{"state": feature, "value": str(value)}]
        tpl = action_index.get(action_b, {})
        if tpl.get("type") == "control":
            # "power_button.set" -> property="power_button", action="set"
            parts = action_b.rsplit(".", 1)
            prop  = parts[0] if len(parts) == 2 else action_b
            act   = parts[1] if len(parts) == 2 else action_b
            return [{"action":   act,
                     "target":   {"type": "instrument", "id": f"${var_name}"},
                     "property": prop,
                     "value":    str(value)}]
        else:
            return [{"action":   action_b,
                     "target":   {"type": "instrument", "id": f"${var_name}"},
                     "property": feature,
                     "value":    str(value)}]

    output_rules = []
    seen_export  = set()
    rule_id      = 1

    for (action_b, feature, value, action_a, feat, forbidden) in sorted(
            precedence_rules, key=lambda x: (x[3], x[1], str(x[0]))):
        key = (action_b, feature, value, action_a)
        if key in seen_export:
            continue
        seen_export.add(key)
        if action_a not in action_index:
            continue
        output_rules.append({
            "id":       rule_id,
            "trigger":  make_trigger(action_a),
            "requires": make_requires(action_b, feature, value)
        })
        rule_id += 1

    out_path = args.input_file.replace(".json", "_rules.json")
    with open(out_path, "w") as f:
        json.dump(output_rules, f, indent=2)
    print(f"\nRules exported to: {out_path}")
    print(json.dumps(output_rules, indent=2))
