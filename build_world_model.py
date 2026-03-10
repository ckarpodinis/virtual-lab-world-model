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

    model = {}

    for (s, a), next_states in counts.items():

        total = sum(v["count"] for v in next_states.values())

        transitions = {}
        expected_reward = 0

        for ns, data in next_states.items():

            prob = data["count"] / total
            avg_reward = data["reward_sum"] / data["count"]

            transitions[ns] = {
                "probability": prob,
                "count": data["count"],
                "reward_sum": data["reward_sum"],
                "avg_reward": avg_reward
            }

            expected_reward += prob * avg_reward

        model[(s, a)] = {
            "expected_reward": expected_reward,
            "transitions": transitions
        }

    return model

def print_model(model):

    for (s, a), data in model.items():

        print("\nSTATE:", decode_state(s))
        print("ACTION:", a)
        print("EXPECTED REWARD:", round(data["expected_reward"], 3))
        print("TRANSITIONS:")

        for ns, tdata in data["transitions"].items():

            print(
                "  ->",
                decode_state(ns),
                f"(p={tdata['probability']:.3f}, "
                f"count={tdata['count']}, "
                f"reward_sum={tdata['reward_sum']}, "
                f"avg_reward={tdata['avg_reward']:.2f})"
            )

def main():

    parser = argparse.ArgumentParser(
        description="Build tabular world model from transition logs"
    )

    parser.add_argument("input", help="JSONL transitions file")

    args = parser.parse_args()

    model = build_world_model(args.input)

    print_model(model)


if __name__ == "__main__":
    main()
