"""Build a possibilistic world model from judged programmatic transitions."""

import argparse
import json
from collections import OrderedDict


def _filtered_state(state, ignore_features):
    return {key: value for key, value in state.items() if key not in ignore_features}


def build_possibilistic_world_model(filename, ignore_features, threshold):
    groups = OrderedDict()
    with open(filename, encoding="utf-8") as handle:
        for line in handle:
            transition = json.loads(line)
            state = _filtered_state(transition["state"], ignore_features)
            next_state = _filtered_state(transition["next_state"], ignore_features)
            action = transition["action"]
            reward = transition.get("reward", 0)
            key = json.dumps([state, action], sort_keys=True, separators=(",", ":"))
            group = groups.setdefault(key, {
                "state": state, "action": action, "rewards": [], "transitions": []
            })
            group["rewards"].append(reward)
            group["transitions"].append({"next_state": next_state, "reward": reward})

    model = []
    for group in groups.values():
        possibility = max(group["rewards"])
        model.append({
            "state": group["state"],
            "action": group["action"],
            "possibility": possibility,
            "applicable": possibility >= threshold,
            "transitions": group["transitions"],
        })
    return model


def main():
    parser = argparse.ArgumentParser(description="Build a possibilistic world model")
    parser.add_argument("input", help="JSONL transitions file")
    parser.add_argument("-o", "--output", default="world_model.json")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--ignore-features", nargs="+", default=[], metavar="FEATURE")
    args = parser.parse_args()
    model = build_possibilistic_world_model(
        args.input, set(args.ignore_features), args.threshold
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2)
    print(f"Built {len(model)} possibilistic (state, action) entries")
    print(f"World model saved to {args.output}")


if __name__ == "__main__":
    main()
