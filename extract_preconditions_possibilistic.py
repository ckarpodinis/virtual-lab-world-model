"""Extract single-variable rules from a possibilistic world model."""

import argparse
import json
from collections import defaultdict
from itertools import combinations


def parse_world_model(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = [data]
    return [{
        "state": item["state"],
        "action": item["action"],
        "possibility": item["possibility"],
        "applicable": item["applicable"],
        "transitions": item.get("transitions", []),
    } for item in data]


def _forbidden_contrasts(entries_a):
    contrasts = defaultdict(set)
    for first, second in combinations(entries_a, 2):
        if first["applicable"] == second["applicable"]:
            continue
        if first["state"].keys() != second["state"].keys():
            continue
        differences = [key for key in first["state"]
                       if first["state"][key] != second["state"][key]]
        if len(differences) == 1:
            key = differences[0]
            inapplicable = first if not first["applicable"] else second
            contrasts[key].add(inapplicable["state"][key])
    return contrasts


def extract_rules(entries_a):
    applicable = [entry for entry in entries_a if entry["applicable"]]
    if not applicable:
        return {"note": "always inapplicable regardless of state"}
    contrasts = _forbidden_contrasts(entries_a)
    rules = {}
    for key in sorted(entries_a[0]["state"]):
        values = {entry["state"][key] for entry in entries_a}
        result = {}
        for value in sorted(values, key=str):
            absent = not any(e["applicable"] and e["state"][key] == value
                             for e in entries_a)
            necessary = all(e["state"][key] == value for e in applicable)
            forbidden = absent and value in contrasts[key]
            result[value] = ("NECESSARY" if necessary else
                             "FORBIDDEN" if forbidden else "NEUTRAL")
        rules[key] = result
    if (all(e["applicable"] for e in entries_a)
            and not any(role != "NEUTRAL" for r in rules.values() for role in r.values())):
        return {"note": "always applicable regardless of state"}
    return rules


def extract_preconditions(entries):
    by_action = defaultdict(list)
    for entry in entries:
        by_action[entry["action"]].append(entry)
    actions = []
    for action in sorted(by_action):
        rules = extract_rules(by_action[action])
        block = {"action": action, "preconditions": []}
        if "note" in rules:
            block["note"] = rules["note"]
        else:
            for variable, value_map in rules.items():
                for value, role in sorted(value_map.items(), key=lambda item: str(item[0])):
                    if role != "NEUTRAL":
                        block["preconditions"].append({
                            "type": role, "variable": variable, "value": value
                        })
        actions.append(block)
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Possibilistic world model JSON")
    parser.add_argument("--template", help="MDP template JSON")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()
    output = {"actions": extract_preconditions(parse_world_model(args.input))}
    if args.template:
        with open(args.template, encoding="utf-8") as handle:
            output = {"object": json.load(handle)["object"], **output}
    output_path = args.output or args.input.replace(".json", "_preconditions.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved structured possibilistic preconditions to {output_path}")


if __name__ == "__main__":
    main()
