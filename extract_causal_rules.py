import json
from collections import defaultdict

def normalize_value(v):
    if isinstance(v, str) and v.startswith("location:"):
        return v[len("location:"):]
    return v

# -------------------------------
# Build producer map
# -------------------------------
def build_producer_map(template):
    producer = {}
    object_id = template["object"]

    for action in template["actions"]:
        name = action["name"]
        params = action.get("parameters", {})

        # --- Control actions ---
        if action.get("type") == "control" and "value" in params:
            values = params["value"] if isinstance(params["value"], list) else [params["value"]]
            for v in values:
                for s in template["states"]:
                    if s["name"] in name:
                        producer[(s["name"], v)] = f"{name}(value={v})"

        # --- Interaction actions ---
        if action.get("type") == "interaction":
            subtype = action.get("subtype")

            if subtype == "place":
                obj = params.get("object")
                tgt = params.get("target")

                if obj and tgt:
                    action_str = f"{name}(object={obj}, target={tgt})"

                    # object.location = target
                    producer[(f"{obj}.location", normalize_value(tgt))] = action_str

                    # if target is also a state variable, target = object
                    state_names = {s["name"] for s in template["states"]}
                    if tgt in state_names:
                        producer[(tgt, normalize_value(obj))] = action_str

            elif subtype == "transfer":
                mat = params.get("material")
                tgt = params.get("target_object")

                if mat and tgt:
                    action_str = f"{name}(material={mat}, target_object={tgt})"

                    # If transfer goes into the object described by this template,
                    # the affected state variable is the local "material"
                    if tgt == object_id:
                        producer[("material", mat)] = action_str
                    else:
                        producer[(f"{tgt}.material", mat)] = action_str

    return producer

# -------------------------------
# Build final rules
# -------------------------------
def build_rules(template, preconditions_json):
    producer_map = build_producer_map(template)

    results = []

    for action_block in preconditions_json["actions"]:
        action = action_block["action"]

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

            producer = producer_map.get((var, val))

            # STRONG if same variable has forbidden values
            strength = "STRONG" if var in forbidden else "WEAK"

            results.append({
                "action": action,
                "precondition": {
                    "variable": var,
                    "value": val
                },
                "producer": producer,
                "strength": strength
            })

    return {"rules": results}

def print_rules(final_rules):
    print("\n" + "=" * 70)
    print("FINAL CAUSAL RULES")
    print("=" * 70)

    by_action = {}

    # group rules by action
    for r in final_rules["rules"]:
        by_action.setdefault(r["action"], []).append(r)

    for action in sorted(by_action):
        print(f"\n▶ Action: {action}")

        for r in by_action[action]:
            strength = r["strength"]
            symbol = "✓ STRONG" if strength == "STRONG" else "⚠ WEAK"

            var = r["precondition"]["variable"]
            val = r["precondition"]["value"]
            producer = r["producer"]

            print(f"\n  {symbol}")
            print(f"    requires: {var} = '{val}'")

            if producer:
                print(f"    produced by → {producer}")
            else:
                print(f"    produced by → (no known producer)")

        print("\n" + "-" * 70)

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("template")
    parser.add_argument("preconditions")
    parser.add_argument("-o", "--output", default="final_rules.json")

    args = parser.parse_args()

    with open(args.template) as f:
        template = json.load(f)

    with open(args.preconditions) as f:
        preconditions = json.load(f)

    final = build_rules(template, preconditions)
    
    print_rules(final)

    with open(args.output, "w") as f:
        json.dump(final, f, indent=2)
    
    print(f"Saved final rules to {args.output}")
