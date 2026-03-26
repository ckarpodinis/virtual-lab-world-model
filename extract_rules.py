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
from collections import defaultdict, Counter

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="Path to world model JSON file")
parser.add_argument("--threshold", type=float, default=0.7,
                    help="Reward threshold above which an action is considered VALID (default: 0.7)")
parser.add_argument("--template", help="Path to MDP template JSON (enables JSON rule export)")
parser.add_argument("--epsilon", type=float, default=0.1,
                    help="Ambiguity band below threshold (default: 0.1)")
parser.add_argument("--print-json", action="store_true",
                    help="Export rules to JSON file")
args = parser.parse_args()

VALID_THRESHOLD   = args.threshold
EPSILON           = args.epsilon
INVALID_THRESHOLD = VALID_THRESHOLD - EPSILON

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
    elif reward >= invalid_thresh:
        return "AMBIGUOUS"
    else:
        return "INVALID"

for e in entries:
    e["label"] = classify(e["expected_reward"], VALID_THRESHOLD, INVALID_THRESHOLD)

by_action = defaultdict(list)
for e in entries:
    by_action[e["action"]].append(e)

# ── 3. Validity table ─────────────────────────────────────────────────────────
print("=" * 70)
print(f"VALIDITY TABLE  (VALID >= {VALID_THRESHOLD}, AMBIGUOUS >= {INVALID_THRESHOLD:.2f}, INVALID < {INVALID_THRESHOLD:.2f})")
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

def extract_probabilistic_preconditions(entries_a):
    """
    For each feature, compute how often each value appears among VALID entries.
    confidence = count(value among VALID states) / number of VALID states

    confidence == 1.0  -> logically NECESSARY
    confidence < 1.0   -> probabilistic tendency
    """
    valid = [e for e in entries_a if e["label"] == "VALID"]

    if not valid:
        return {"note": "no valid entries"}

    state_keys = sorted(entries_a[0]["state"].keys())
    result = {}

    for key in state_keys:
        values = [e["state"][key] for e in valid]
        counts = Counter(values)
        total = len(valid)

        result[key] = []
        for val, count in sorted(counts.items(), key=lambda x: (-x[1], str(x[0]))):
            confidence = count / total
            result[key].append({
                "value": val,
                "count": count,
                "total": total,
                "confidence": confidence
            })

    return result

def extract_probabilistic_preconditions_weighted(entries_a):
    valid = [e for e in entries_a if e["label"] == "VALID"]

    if not valid:
        return {"note": "no valid entries"}

    state_keys = sorted(entries_a[0]["state"].keys())
    result = {}

    for key in state_keys:
        weighted_counts = {}
        total_weight = 0

        for e in valid:
            val = e["state"][key]

            weight = sum(t["count"] for t in e.get("transitions", []))
            if weight == 0:
                weight = 1

            weighted_counts[val] = weighted_counts.get(val, 0) + weight
            total_weight += weight

        result[key] = []
        for val, wcount in sorted(weighted_counts.items(), key=lambda x: (-x[1], str(x[0]))):
            result[key].append({
                "value": val,
                "count": wcount,
                "total": total_weight,
                "confidence": wcount / total_weight
            })

    return result

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
    prob_unweighted = extract_probabilistic_preconditions(entries_a)
    prob_weighted   = extract_probabilistic_preconditions_weighted(entries_a)

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
    
    if "note" not in prob_unweighted:
        print(f"  ~ PROBABILISTIC (state-based):")
        for key, items in prob_unweighted.items():
            vals = []
            for item in items:
                vals.append(
                    f"{key}={repr(item['value'])} "
                    f"(conf={item['confidence']:.2f}, {item['count']}/{item['total']})"
                )
            print("    " + " ; ".join(vals))
    if "note" not in prob_weighted:
        print(f"  ~ PROBABILISTIC (experience-weighted):")
        for key, items in prob_weighted.items():
            vals = []
            for item in items:
                vals.append(
                    f"{key}={repr(item['value'])} "
                    f"(conf={item['confidence']:.2f}, {item['count']}/{item['total']})"
                )
            print("    " + " ; ".join(vals))

# ── 6. Precedence rules ───────────────────────────────────────────────────────
# Reuses SUMMARY OF RULES results directly:
#   Step 1 — for each NECESSARY feature=V on action A,
#             confirm that feature!=V always results in INVALID
#             (i.e. the negation is always bad — it's a true precondition).
#   Step 2 — find all actions (excluding self-loops) that produce
#             feature=V via a transition with avg_reward >= threshold.
#   Step 3 — group by action_a: collect all producer actions across all
#             NECESSARY features, deduplicate, and emit one line per producer.
#             Features with no known producer are reported separately.

print("\n" + "=" * 70)
print("FINAL RULES")
print("=" * 70)

# Build producer lookup: (feature, value) -> set of actions that produce it.
# Only keep producers with avg_reward >= VALID_THRESHOLD (trusted transitions).
producer_map = defaultdict(set)
for e in entries:
    for t in e["transitions"]:
        if t["avg_reward"] < VALID_THRESHOLD:
            continue
        for feature, new_val in t["next_state"].items():
            old_val = e["state"].get(feature)
            if old_val != new_val:   # actual state change only
                producer_map[(feature, new_val)].add(e["action"])

any_rule = False

# precedence_data collects structured info for section 7 JSON export:
# list of (action_b_or_None, feature, nec_val, action_a)
precedence_data = []

for action_a in sorted(by_action):
    entries_a = by_action[action_a]
    rules_a   = extract_rules(entries_a)

    if "note" in rules_a:
        continue

    # Accumulate all trusted producer actions across every NECESSARY feature,
    # and separately track features with no known producer.
    all_producers  = set()
    no_producer_fv = []   # (feature, nec_val) pairs with no known producer

    for feature, val_map in rules_a.items():
        necessary_vals = [v for v, role in val_map.items() if role == "NECESSARY"]

        for nec_val in necessary_vals:
            # Step 1: confirm negation — every entry where feature != nec_val
            #         must be INVALID (true precondition check)
            negations = [e for e in entries_a if e["state"][feature] != nec_val]
            if not negations:
                continue
            if any(e["label"] == "VALID" for e in negations):
                continue   # negation is not always INVALID — not a true precondition

            # Step 2: find all trusted producers, exclude self-loops
            producers = sorted(producer_map.get((feature, nec_val), set()) - {action_a})

            if producers:
                all_producers.update(producers)
                for p in producers:
                    precedence_data.append((p, feature, nec_val, action_a))
            else:
                no_producer_fv.append((feature, nec_val))
                precedence_data.append((None, feature, nec_val, action_a))

    # Step 3: emit one rule per producer action, plus any no-producer entries
    for producer in sorted(all_producers):
        print(f"\n  '{producer}'  must precede  '{action_a}'")
        any_rule = True

    for feature, nec_val in no_producer_fv:
        print(f"\n  '{feature}={repr(nec_val)}' (no known producer)  must precede  '{action_a}'")
        any_rule = True

if not any_rule:
    print("\n  No precedence rules found with current threshold.")
print("\n")

# ── 7. Export rules to JSON (only when --template is provided) ────────────────
if args.template:
    import re

    with open(args.template) as f:
        tmpl = json.load(f)

    obj_id       = tmpl["object"]                           # "instrument:electronic_scale[0]"
    action_index = {a["name"]: a for a in tmpl["actions"]}

    def lookup_action(action_name):
        """Match world-model action names (may include params) to template entries.
        e.g. 'place(object=tool:aluminum_foil, target=weighing_platform)' -> 'place'
        """
        if action_name in action_index:
            return action_index[action_name]
        base = action_name.split("(")[0]   # strip parameters
        return action_index.get(base, {})

    # instrument id in dot notation for trigger: "instrument.electronic_scale[0]"
    obj_dot  = obj_id.replace(":", ".")

    def make_trigger(action_a):
        tpl    = lookup_action(action_a)
        params = tpl.get("parameters", {})
        conds  = [{"path": "action", "equals": action_a.split("(")[0]}]
        if tpl.get("type") == "interaction":
            obj_param = params.get("object", "")
            if obj_param:
                obj_param = obj_param.replace(":", ".")
                if "[" in obj_param:
                    conds.append({"path": "target.id", "equals": obj_param})
                else:
                    conds.append({"path": "target.id", "matches": {"glob": obj_param + ".*"}})
            target_param = params.get("target", "")
            if target_param:
                conds.append({"path": "to.id", "equals": obj_dot + "." + target_param})
            else:
                conds.append({"path": "to.id", "equals": obj_dot})
        elif tpl.get("type") == "control":
            base = action_a.split("(")[0]
            parts = base.rsplit(".", 1)

            if len(parts) == 2:
                prop, act = parts
                conds[0] = {"path": "action", "equals": act}  # replace action
                conds.append({"path": "target.id", "equals": obj_dot + "." + prop})
            else:
                conds.append({"path": "target.id", "equals": obj_dot})

        return {"match": {"all": conds}}

    def make_requires(action_b, feature, value):
        if action_b is None:
            # no trusted producer — express as a state condition
            return [{"state": feature, "value": str(value)}]
        tpl = lookup_action(action_b)
        if tpl.get("type") == "control":
            # "power_button.set(value=on)" -> base "power_button.set" -> property="power_button", action="set"
            base_b = action_b.split("(")[0]
            parts = base_b.rsplit(".", 1)

            if len(parts) == 2:
                prop, act = parts
                target_id = obj_dot + "." + prop
            else:
                act = action_b
                target_id = obj_dot

            return [{
                "action": act,
                "target": {
                    "id": target_id
                },
                "value": str(value)
            }]
        else:
            return [{
                "action": action_b,
                "target": {
                    "id": obj_dot + "." + feature
                },
                "value": str(value)
            }]

    output_rules = []
    seen_export  = set()
    rule_id      = 1

    for (action_b, feature, value, action_a) in sorted(
            precedence_data, key=lambda x: (x[3], x[1], str(x[0]))):
        # deduplicate on (action_b, action_a) — same producer+trigger pair
        # regardless of which feature caused the rule
        key = (action_b, action_a)
        if key in seen_export:
            continue
        seen_export.add(key)
        if not lookup_action(action_a):
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
    if args.print_json:
        print(json.dumps(output_rules, indent=2))

# ── 8. Graphical visualization ────────────────────────────────────────────────
if 'output_rules' in locals():
    try:
        from graphviz import Digraph

        dot = Digraph(comment="Rule Graph")
        dot.attr(rankdir="LR")

        # ── Extract trigger (action + target) ─────────────────────────────────
        def extract_trigger(trigger):
            action = None
            target = None

            for p in trigger["match"]["all"]:
                if p["path"] == "action":
                    action = p["equals"]
                elif p["path"] == "target.id":
                    target = p.get("equals") or p.get("matches", {}).get("glob")

            return action, target

        # ── Node helpers (FULL identity = action + target + value) ────────────
        def node_id(action, target, value):
            return f"{action}|{target}|{value}"

        def node_label(action, target, value):
            return f"{action}\n{target}\n= {value}"

        # ── Build graph ───────────────────────────────────────────────────────
        for rule in output_rules:
            action, target = extract_trigger(rule["trigger"])

            # collect all values for this action-target pair
            values = set()
            for r in output_rules:
                a2, t2 = extract_trigger(r["trigger"])
                if a2 == action and t2 == target:
                    for req2 in r["requires"]:
                        if "value" in req2:
                            values.add(req2["value"])

            for req in rule["requires"]:
                if "action" not in req:
                    continue

                r_action = req["action"]
                r_target = req.get("target", {}).get("id", "")
                r_value  = req.get("value", "")

                r_id = node_id(r_action, r_target, r_value)
                dot.node(r_id, label=node_label(r_action, r_target, r_value))

                # connect to OTHER values (THIS is the fix)
                if len(values) > 1:
                    # multi-valued → connect to other values
                    for v in values:
                        if v == r_value:
                            continue

                        t_id = node_id(action, target, v)
                        dot.node(t_id, label=node_label(action, target, v))
                        dot.edge(r_id, t_id)
                else:
                    # single-valued → connect directly to trigger node
                    t_id = node_id(action, target, r_value)
                    dot.node(t_id, label=node_label(action, target, r_value))
                    dot.edge(r_id, t_id)
        
        # ── Save graph using JSON filename ────────────────────────────────────
        graph_path = out_path.replace("_rules.json", "_rules_graph")
        dot.render(graph_path, format="png", cleanup=True)

        print(f"Graph saved to: {graph_path}.png")

    except ImportError:
        print("Graphviz not installed. Run: pip install graphviz")
