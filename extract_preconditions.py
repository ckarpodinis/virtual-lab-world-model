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
parser.add_argument("input", help="Path to world model JSON file")
parser.add_argument("--threshold", type=float, default=0.7,
                    help="Reward threshold above which an action is considered VALID (default: 0.7)")
parser.add_argument("--template", help="Path to MDP template JSON (enables JSON rule export)")
parser.add_argument("--epsilon", type=float, default=0.1,
                    help="Ambiguity band below threshold (default: 0.1)")
parser.add_argument("--print-json", action="store_true",
                    help="Export rules to JSON file")
parser.add_argument("--conf-threshold", type=float, default=0.9,
                    help="Min weighted confidence for NECESSARY, max for FORBIDDEN (default: 0.9)")
parser.add_argument("-o", "--output", default=None, help="Output JSON file")
args = parser.parse_args()

VALID_THRESHOLD   = args.threshold
EPSILON           = args.epsilon
INVALID_THRESHOLD = 1 - VALID_THRESHOLD
CONF_THRESHOLD     = args.conf_threshold
INVALID_CONF_THRESHOLD = args.conf_threshold

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

entries = parse_world_model(args.input)
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
def short(val):
    s = str(val)
    return s.rsplit(":", 1)[-1] if ":" in s else s

print("=" * 70)
print(f"VALIDITY TABLE  (VALID >= {VALID_THRESHOLD}, AMBIGUOUS >= {INVALID_THRESHOLD:.2f}, INVALID < {INVALID_THRESHOLD:.2f})")
print("=" * 70)
for action in sorted(by_action):
    entries_a  = by_action[action]
    state_keys = sorted(entries_a[0]["state"].keys())
    print(f"\nAction: '{action}'")
    header = "  " + "  ".join(f"{short(k):<30}" for k in state_keys) + f"  {'reward':<8}  label"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for e in sorted(entries_a, key=lambda x: str(x["state"])):
        vals      = "  ".join(f"{short(e['state'][k]):<30}" for k in state_keys)
        label_sym = {"VALID": "✓", "INVALID": "✗", "AMBIGUOUS": "?"}[e["label"]]
        print(f"  {vals}  {e['expected_reward']:<8.3f}  {label_sym} {e['label']}")

# ── 4. Rule extraction ────────────────────────────────────────────────────────
def extract_rules(entries_a):
    """
    For each state feature, classify each value using experience-weighted confidence:

      NECESSARY  — this value has weighted confidence >= CONF_THRESHOLD among valid
                   entries, AND all other values have confidence <= (1-CONF_THRESHOLD).
                   This means the feature is essentially binary-like and one value
                   dominates valid entries cleanly.

      FORBIDDEN  — this value never appears in any valid entry (conf == 0.0).

      NEUTRAL    — everything else (multi-valued features like quantity, or values
                   that appear in both valid and invalid entries).
    """
    
    VALID_ABSENCE_THRESHOLD = 0.05

    valid   = [e for e in entries_a if e["label"] == "VALID"]
    invalid = [e for e in entries_a if e["label"] == "INVALID"]
    ambiguous = [e for e in entries_a if e["label"] == "AMBIGUOUS"]

    if not valid and not invalid:
        return {"note": "all entries ambiguous, no clean rules extractable"}
    # Don't short-circuit on "no invalid" — NECESSARY can still be detected
    # from a skewed confidence distribution even when all entries are valid.
    # E.g. cap.close: cap='opened' has conf=0.94, cap='closed' conf=0.06
    # → cap='opened' is NECESSARY even though no invalid entries exist.
    if not valid:
        return {"note": "always invalid regardless of state"}

    state_keys = sorted(entries_a[0]["state"].keys())

    # Compute experience-weighted confidence per feature value among VALID entries
    def weighted_conf(entries_valid, key):
        weighted_counts = {}
        total_weight = 0
        for e in entries_valid:
            val = e["state"][key]
            weight = sum(t["count"] for t in e.get("transitions", [])) or 1
            weighted_counts[val] = weighted_counts.get(val, 0) + weight
            total_weight += weight
        if total_weight == 0:
            return {}
        return {val: w / total_weight for val, w in weighted_counts.items()}

    from itertools import combinations

    causal_score = defaultdict(lambda: defaultdict(int))

    for e1, e2 in combinations(entries_a, 2):

        diffs = [
            k for k in e1["state"]
            if e1["state"][k] != e2["state"][k]
        ]

        if len(diffs) != 1:
            continue

        k = diffs[0]
        v1 = e1["state"][k]
        v2 = e2["state"][k]

        if e1["label"] == "VALID" and e2["label"] == "INVALID":
            causal_score[k][v2] += 1
            causal_score[k][v1] -= 1

        elif e2["label"] == "VALID" and e1["label"] == "INVALID":
            causal_score[k][v1] += 1
            causal_score[k][v2] -= 1

    rules = {}
    for key in state_keys:
        conf_map = weighted_conf(valid, key)
        invalid_conf_map = weighted_conf(invalid, key)
        
        ambiguous_vals = set(e["state"][key] for e in ambiguous)
        invalid_vals = set(e["state"][key] for e in invalid)
        all_vals     = set(e["state"][key] for e in entries_a)

        result = {}
#        for val in sorted(all_vals, key=str):
#            vc = conf_map.get(val, 0.0)
#            cs = causal_score[key].get(val, 0)
#
#            if (
#                vc <= VALID_ABSENCE_THRESHOLD   # not seen in valid
#                and cs > 0                      # causes invalid
#                ):
#                result[val] = "FORBIDDEN"
#            else:
#                result[val] = "NEUTRAL"
        
        for val in sorted(all_vals, key=str):
            vc = conf_map.get(val, 0.0)
            ic = invalid_conf_map.get(val, 0.0)
            cs = causal_score[key].get(val, 0)

            if vc <= VALID_ABSENCE_THRESHOLD:

                if cs > 0:
                    result[val] = "FORBIDDEN"   # causal

                elif ic >= INVALID_CONF_THRESHOLD:
                    result[val] = "FORBIDDEN"   # fallback if not causal

                else:
                    result[val] = "NEUTRAL"

            else:
                result[val] = "NEUTRAL"

        # NECESSARY: exactly one value dominates valid entries (conf >= CONF_THRESHOLD)
        # AND all other values seen in valid have conf <= (1 - CONF_THRESHOLD)
        high_conf_vals = [v for v, c in conf_map.items() if c >= CONF_THRESHOLD]
        if len(high_conf_vals) == 1:
            sole = high_conf_vals[0]
            other_confs = [c for v, c in conf_map.items() if v != sole]
            if all(c <= (1 - CONF_THRESHOLD) for c in other_confs):
                result[sole] = "NECESSARY"

        rules[key] = result

    # Return "always valid" only if no NECESSARY values found and no invalid entries
    has_necessary = any(role == "NECESSARY"
                        for vm in rules.values() for role in vm.values())
    if not invalid and not has_necessary:
        return {"note": "always valid regardless of state"}

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

    if not entries_a:
        return {}
    
    state_keys = sorted(entries_a[0]["state"].keys())
    result = {}

    for key in state_keys:
        weighted_counts = {}
        total_weight = 0

        for e in entries_a:
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

structured_rules = []

for action in sorted(by_action):
    entries_a = by_action[action]
    rules     = extract_rules(entries_a)

    print(f"\nAction: '{action}'")

    action_block = {
        "action": action,
        "preconditions": []
    }

    if "note" in rules:
        print(f"  → {rules['note']}")
        action_block["note"] = rules["note"]
        structured_rules.append(action_block)
        continue

    printed = False

    for key, val_map in rules.items():
        for val, role in sorted(val_map.items(), key=lambda x: (x[1], str(x[0]))):
            if role == "NEUTRAL":
                continue

            sym = {
                "NECESSARY": "⚠ NECESSARY",
                "FORBIDDEN": "✗ FORBIDDEN"
            }[role]

            print(f"  {sym:<16}  {key} = {repr(val)}")
            printed = True

            # ✅ ADD STRUCTURED ENTRY
            action_block["preconditions"].append({
                "type": role,
                "variable": key,
                "value": val
            })

    if not printed:
        print("  → No single-feature NECESSARY or FORBIDDEN values found")
        print("    All features are NEUTRAL — validity depends on feature combinations")

    structured_rules.append(action_block)

if args.output:
    output_path = args.output
else:
    output_path = args.input.replace(".json", "_preconditions.json")

with open(output_path, "w") as f:
    json.dump({"actions": structured_rules}, f, indent=2)

print(f"\nSaved structured preconditions to {output_path}")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY OF RULES")
print("=" * 70)

for action in sorted(by_action):
    entries_a  = by_action[action]
    valid      = [e for e in entries_a if e["label"] == "VALID"]
    ambig      = [e for e in entries_a if e["label"] == "AMBIGUOUS"]
    invalid    = [e for e in entries_a if e["label"] == "INVALID"]
    state_keys = sorted(entries_a[0]["state"].keys())
    
    rules      = extract_rules(entries_a)
    
    valid_prob_weighted   = extract_probabilistic_preconditions_weighted(valid)
    invalid_prob_weighted = extract_probabilistic_preconditions_weighted(invalid)

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
    
#    if "note" not in prob_unweighted:
#        print(f"  ~ PROBABILISTIC (state-based):")
#        for key, items in prob_unweighted.items():
#            vals = []
#            for item in items:
#                vals.append(
#                    f"{key}={repr(item['value'])} "
#                    f"(conf={item['confidence']:.2f}, {item['count']}/{item['total']})"
#                )
#            print("    " + " ; ".join(vals))
    print(f"  ~ PROBABILISTIC for VALID (experience-weighted):")
    for key, items in valid_prob_weighted.items():
        vals = []
        for item in items:
            vals.append(
                f"{key}={repr(item['value'])} "
                f"(conf={item['confidence']:.2f}, {item['count']}/{item['total']})"
            )
        print("    " + " ; ".join(vals))
    print(f"  ~ PROBABILISTIC for INVALID (experience-weighted):")
    for key, items in invalid_prob_weighted.items():
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
            negation_vals = [v for v, role in val_map.items() if v != nec_val]
            
            if negation_vals and all(val_map.get(v) == "FORBIDDEN" for v in negation_vals):
                strength = "strong"
            else:
                strength = "weak"

            # proceed to producer lookup...
            producers = sorted(producer_map.get((feature, nec_val), set()) - {action_a})

            if producers:
                for p in producers:
                    precedence_data.append((p, feature, nec_val, action_a, strength))
            else:
                precedence_data.append((None, feature, nec_val, action_a, strength))

seen = set()
for (action_b, feature, nec_val, action_a, strength) in sorted(
        precedence_data, key=lambda x: (x[3], str(x[0]))):
    key = (action_b, action_a)
    if key in seen:
        continue
    seen.add(key)
    if action_b:
        if strength == "strong":
            print(f"\n  ✓ STRONG:  '{action_b}' must precede '{action_a}'")
        else:
            print(f"\n  ⚠ WEAK:    '{action_b}' must precede '{action_a}'")
    else:
        if strength == "strong":
            print(f"\n  ✓ STRONG:  '{feature}={repr(nec_val)}' (no known producer)  must precede  '{action_a}'")
        else:
            print(f"\n  ⚠ WEAK:    '{feature}={repr(nec_val)}' (no known producer)  must precede  '{action_a}'")
    any_rule = True

if not any_rule:
    print("\n  No precedence rules found with current threshold.")
print("\n")

# ── 7. Export rules to JSON (only when --template is provided) ────────────────
if args.template:
    import re

    with open(args.template) as f:
        tmpl = json.load(f)

    obj_id       = tmpl["object"]
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

    out_path = args.input.replace(".json", "_rules.json")
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
            raw = f"{action}|{target}|{value}"
            return re.sub(r'[^a-zA-Z0-9_]', '_', raw)

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
