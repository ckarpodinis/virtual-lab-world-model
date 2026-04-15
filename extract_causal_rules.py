import json
import glob
import re
from collections import defaultdict


def normalize_value(v):
    if isinstance(v, str) and v.startswith("location:"):
        return v[len("location:"):]
    return v


def strip_instance_index(object_id):
    """
    "container:ddh2o_bottle[0]"  ->  "container:ddh2o_bottle"
    "tool:erlenmeyer_flask"      ->  "tool:erlenmeyer_flask"   (unchanged)
    """
    return re.sub(r'\[\d+\]$', '', object_id)


def is_bare_var(var):
    """
    Bare variable: no '.' and no ':', e.g. "power_button", "material".
    Qualified variable: already names its owner, e.g. "container:ddh2o_bottle.cap".
    """
    return '.' not in var and ':' not in var


def normalize_qualified_var(var):
    """
    Strip the instance index from the object part of a qualified variable.
    e.g. "instrument:electronic_pipette[0].material" -> "instrument:electronic_pipette.material"
         "container:ddh2o_bottle.cap"                -> "container:ddh2o_bottle.cap"  (unchanged)
    """
    dot = var.rfind('.')
    if dot == -1:
        return var
    obj_part   = var[:dot]
    state_part = var[dot:]          # includes the leading '.'
    return strip_instance_index(obj_part) + state_part


# -----------------------------------------------------------------------
# Build producer map for ONE template
# -----------------------------------------------------------------------
def build_producer_map(template):
    """
    Returns:  { (qualified_var, value) -> {"action": str, "source": object_id} }

    All keys are stored in qualified form "object_id_base.state_name" to
    prevent collisions when multiple templates share bare state names.
    """
    producer = {}
    object_id      = template["object"]
    object_id_base = strip_instance_index(object_id)
    source_label   = object_id

    for action in template["actions"]:
        name   = action["name"]
        params = action.get("parameters", {})

        # --- Control actions ---
        if action.get("type") == "control" and "value" in params:
            values = params["value"] if isinstance(params["value"], list) else [params["value"]]
            for v in values:
                for s in template["states"]:
                    if s["name"] in name:
                        entry = {
                            "action": f"{name}(value={v})",
                            "source": source_label,
                        }
                        qualified = f"{object_id_base}.{s['name']}"
                        producer[(qualified, v)] = entry

        # --- Interaction actions ---
        if action.get("type") == "interaction":
            subtype = action.get("subtype")

            if subtype == "place":
                obj = params.get("object")
                tgt = params.get("target")
                if obj and tgt:
                    action_str = f"{name}(object={obj}, target={tgt})"

                    # Key 1: obj.location = where it was placed.
                    # If tgt is a bare state name (e.g. "stirring_platform"), the
                    # actual location value is qualified as "object_id_base:tgt"
                    # (e.g. "instrument:magnetic_stirrer[0]:stirring_platform").
                    # If tgt already looks like a full id (contains ':'), use as-is.
                    state_names = {s["name"] for s in template["states"]}
                    if tgt in state_names:
                        location_val = f"{object_id}:{tgt}"
                    else:
                        location_val = normalize_value(tgt)
                    producer[(f"{obj}.location", location_val)] = {
                        "action": action_str,
                        "source": source_label,
                    }

                    # Key 2: the target state variable reflects which object was placed.
                    # tgt is a bare state name owned by this template → qualify it.
                    if tgt in state_names:
                        producer[(f"{object_id_base}.{tgt}", normalize_value(obj))] = {
                            "action": action_str,
                            "source": source_label,
                        }

            elif subtype == "transfer":
                mat = params.get("material")
                tgt = params.get("target_object")
                if mat and tgt:
                    action_str = f"{name}(material={mat}, target_object={tgt})"
                    if tgt == object_id:
                        producer[(f"{object_id_base}.material", mat)] = {
                            "action": action_str,
                            "source": source_label,
                        }
                    else:
                        producer[(f"{strip_instance_index(tgt)}.material", mat)] = {
                            "action": action_str,
                            "source": source_label,
                        }

    return producer


# -----------------------------------------------------------------------
# Load templates into two maps: own (subject match) and other
# -----------------------------------------------------------------------
def build_producer_maps(template_paths, subject_object):
    """
    Splits templates into two producer maps based on whether the template's
    object matches the preconditions subject object:

      own_map   — the template whose object id equals subject_object
      other_map — all remaining templates merged together

    The subject_object comparison ignores instance index, so
    "instrument:electronic_pipette[0]" matches "instrument:electronic_pipette[0]".
    """
    own_map   = {}
    other_map = {}

    for path in template_paths:
        with open(path) as f:
            template = json.load(f)

        label = template["object"]
        pmap  = build_producer_map(template)

        if label == subject_object:
            own_map.update(pmap)
            print(f"  [own]   loaded {label}  ({len(pmap)} producer keys)")
        else:
            other_map.update(pmap)
            print(f"  [other] loaded {label}  ({len(pmap)} producer keys)")

    print(f"  own map:   {len(own_map)} keys  |  other map: {len(other_map)} keys")
    return own_map, other_map


# -----------------------------------------------------------------------
# Build final rules
# -----------------------------------------------------------------------
def build_rules(own_map, other_map, preconditions_json):
    """
    Lookup order for every precondition variable:

    Qualified var (e.g. "container:ddh2o_bottle.cap"):
        1. own_map    — in case the subject template cross-references itself
        2. other_map  — the foreign template that owns that object

    Bare var (e.g. "power_button", "material"):
        Always refers to the subject object's own state, so:
        1. own_map   qualified with subject_base  (e.g. "instrument:electronic_pipette.power_button")
        2. other_map qualified with subject_base  (fallback — should rarely fire)
        3. other_map qualified with target_base   (cross-object bare var, last resort)

    own_map is always consulted before other_map, so the subject's own template
    always wins when both maps could theoretically answer the same key.
    """
    subject_object = preconditions_json.get("object", "")
    subject_base   = strip_instance_index(subject_object) if subject_object else None

    results = []

    for action_block in preconditions_json["actions"]:
        action = action_block["action"]

        # target_object from the action string (last-resort for cross-object bare vars)
        target_base = None
        m = re.search(r'target_object=([^\s,)]+)', action)
        if m:
            target_base = strip_instance_index(m.group(1))

        necessary = []
        forbidden = defaultdict(set)

        for p in action_block.get("preconditions", []):
            if p["type"] == "NECESSARY":
                necessary.append(p)
            elif p["type"] == "FORBIDDEN":
                forbidden[p["variable"]].add(p["value"])

        for n in necessary:
            var = n["variable"]
            val = normalize_value(n["value"])

            entry = None

            if not is_bare_var(var):
                # Qualified var: own_map first, then other_map.
                # Also try with instance index stripped from the object part,
                # because producer keys are always stored without instance index
                # (e.g. "instrument:electronic_pipette.material") while the
                # precondition variable may carry one
                # (e.g. "instrument:electronic_pipette[0].material").
                norm_var = normalize_qualified_var(var)
                entry = (own_map.get((var, val))
                         or own_map.get((norm_var, val))
                         or other_map.get((var, val))
                         or other_map.get((norm_var, val)))
            else:
                # Bare var: always resolve against subject_base, own_map first
                if subject_base:
                    qvar = f"{subject_base}.{var}"
                    entry = own_map.get((qvar, val)) or other_map.get((qvar, val))
                # Last resort: try target_object in other_map
                if entry is None and target_base and target_base != subject_base:
                    qvar = f"{target_base}.{var}"
                    entry = other_map.get((qvar, val))

            producer_action = entry["action"] if entry else None
            producer_source = entry["source"] if entry else None

            strength = "STRONG" if var in forbidden else "WEAK"

            results.append({
                "action": action,
                "precondition": {"variable": var, "value": val},
                "producer": producer_action,
                "producer_source": producer_source,
                "strength": strength,
            })

    return {"rules": results}


def print_rules(final_rules):
    print("\n" + "=" * 70)
    print("FINAL CAUSAL RULES")
    print("=" * 70)

    by_action = {}
    for r in final_rules["rules"]:
        by_action.setdefault(r["action"], []).append(r)

    for action in sorted(by_action):
        print(f"\n▶ Action: {action}")

        for r in by_action[action]:
            symbol = "✓ STRONG" if r["strength"] == "STRONG" else "⚠ WEAK"
            var    = r["precondition"]["variable"]
            val    = r["precondition"]["value"]
            prod   = r["producer"]
            src    = r.get("producer_source")

            print(f"\n  {symbol}")
            print(f"    requires: {var} = '{val}'")
            if prod:
                src_str = f"  [{src}]" if src else ""
                print(f"    produced by → {prod}{src_str}")
            else:
                print(f"    produced by → (no known producer)")

        print("\n" + "-" * 70)


# -----------------------------------------------------------------------
# Resolve template paths (plain paths + globs)
# -----------------------------------------------------------------------
def resolve_template_paths(raw_args):
    resolved = []
    seen = set()
    for pattern in raw_args:
        matches = sorted(glob.glob(pattern))
        if not matches:
            matches = [pattern]
        for p in matches:
            if p not in seen:
                seen.add(p)
                resolved.append(p)
    return resolved


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Extract causal rules from one or more MDP templates.\n\n"
            "The preconditions file must contain an 'object' field (produced by\n"
            "extract_preconditions.py --template <mdp_template.json>).\n\n"
            "Usage examples:\n"
            "  python extract_causal_rules.py template.json preconditions.json\n"
            "  python extract_causal_rules.py t1.json t2.json preconditions.json\n"
            "  python extract_causal_rules.py 'output9/*_mdp_template.json' preconditions.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files", nargs="+", metavar="FILE",
        help="One or more MDP template JSON files (or glob patterns), "
             "then the preconditions JSON file as the last argument.",
    )
    parser.add_argument("-o", "--output", default="final_rules.json")

    args = parser.parse_args()

    if len(args.files) < 2:
        parser.error("Provide at least one template file and one preconditions file.")

    preconditions_path = args.files[-1]
    template_paths     = resolve_template_paths(args.files[:-1])

    if not template_paths:
        parser.error("No template files matched the provided patterns.")

    with open(preconditions_path) as f:
        preconditions = json.load(f)

    if "object" not in preconditions:
        parser.error(
            f"'{preconditions_path}' has no 'object' field.\n"
            "Re-generate it with: extract_preconditions.py --template <mdp_template.json>"
        )

    subject = preconditions["object"]
    print(f"\nSubject object: {subject}")
    print(f"Loading {len(template_paths)} template file(s)...")

    own_map, other_map = build_producer_maps(template_paths, subject)

    if not own_map:
        print(f"  [warning] no template matched subject '{subject}' — "
              f"all templates treated as 'other'")

    final = build_rules(own_map, other_map, preconditions)

    print_rules(final)

    with open(args.output, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nSaved final rules to {args.output}")
