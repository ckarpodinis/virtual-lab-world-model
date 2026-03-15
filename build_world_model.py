import json
import argparse
from collections import defaultdict


def encode_state(state_dict):
    return tuple(sorted(state_dict.items()))


def decode_state(state_tuple):
    return dict(state_tuple)


def build_world_model(filename):

    counts = defaultdict(lambda: defaultdict(lambda: {"count": 0, "reward_sum": 0}))

    with open(filename) as f:
        for line in f:
            t = json.loads(line)

            s = encode_state(t["state"])
            a = t["action"]
            ns = encode_state(t["next_state"])
            r = t.get("reward", 0)

            counts[(s, a)][ns]["count"] += 1
            counts[(s, a)][ns]["reward_sum"] += r

    model = []

    for (s, a), next_states in counts.items():

        total = sum(v["count"] for v in next_states.values())

        transitions = []
        expected_reward = 0

        for ns, data in next_states.items():

            prob = data["count"] / total
            avg_reward = data["reward_sum"] / data["count"]

            expected_reward += prob * avg_reward

            transitions.append({
                "next_state": decode_state(ns),
                "probability": prob,
                "count": data["count"],
                "reward_sum": data["reward_sum"],
                "avg_reward": avg_reward
            })

        model.append({
            "state": decode_state(s),
            "action": a,
            "expected_reward": expected_reward,
            "transitions": transitions
        })

    return model


def print_model(model):

    for entry in model:

        print("\nSTATE:", entry["state"])
        print("ACTION:", entry["action"])
        print("EXPECTED REWARD:", round(entry["expected_reward"], 3))
        print("TRANSITIONS:")

        for t in entry["transitions"]:
            print(
                "  ->",
                t["next_state"],
                f"(p={t['probability']:.3f}, "
                f"count={t['count']}, "
                f"reward_sum={t['reward_sum']}, "
                f"avg_reward={t['avg_reward']:.2f})"
            )


def save_model(model, outfile):

    with open(outfile, "w") as f:
        json.dump(model, f, indent=2)

    print(f"\nWorld model saved to {outfile}")


def main():

    parser = argparse.ArgumentParser(
        description="Build tabular world model from transition logs"
    )

    parser.add_argument(
        "input",
        help="JSONL transitions file"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="world_model.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    model = build_world_model(args.input)

    # print to console
    print_model(model)

    # save to file
    save_model(model, args.output)


if __name__ == "__main__":
    main()
