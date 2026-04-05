import uuid
import json
import argparse
from itertools import product
from openai import OpenAI

client = OpenAI()

def build_valid_actions(template):
    """Build the set of canonical valid action strings from the template.
    Expands list-valued parameters into all valid combinations."""
    valid = set()
    for action in template["actions"]:
        name = action["name"]
        params = action.get("parameters", {})
        if not params:
            valid.add(f"{name}()")
            continue
        param_keys = list(params.keys())
        param_vals = [v if isinstance(v, list) else [v] for v in params.values()]
        for combo in product(*param_vals):
            parts = ", ".join(f"{k}={v}" for k, v in zip(param_keys, combo))
            valid.add(f"{name}({parts})")
    return valid

def validate_state(state, template):
    """Check that a state assigns valid values to all template state variables.
    Returns (is_valid, reason_or_None)."""
    for var in template["states"]:
        name = var["name"]
        if name not in state:
            return False, f"missing variable '{name}'"
        if var["kind"] == "enum" and state[name] not in var["values"]:
            return False, f"'{name}'={state[name]!r} not in {var['values']}"
    return True, None

def validate_transition(t, valid_actions, template):
    """Return (is_valid, reason) for a generated transition."""
    if t.get("action") not in valid_actions:
        return False, f"unknown action: {t.get('action')!r}"
    ok, reason = validate_state(t.get("state", {}), template)
    if not ok:
        return False, f"invalid state — {reason}"
    ok, reason = validate_state(t.get("next_state", {}), template)
    if not ok:
        return False, f"invalid next_state — {reason}"
    return True, None

SYSTEM_PROMPT = """
You generate state-transition data for Markov Decision Processes (MDPs)
describing laboratory instruments or physical systems.

You are given an MDP template that defines:

- state variables and their domains
- actions that can be performed
- optional effects of actions

Your task is to imagine transitions using your general knowledge and
common sense about how such systems behave.

You are not a deterministic simulator.

First generate a transition (state, action, next_state).
Then evaluate its plausibility and assign a reward.

reward = 1 if the transition appears logically plausible.
reward = 0 if the transition appears inconsistent or unrealistic.

Do not attempt to enforce perfect rules. Imperfect reasoning is allowed.
Mistakes should resemble natural misunderstandings rather than random noise.
"""

USER_PROMPT = """
You are given an MDP template describing an object.

MDP TEMPLATE
------------
{MDP_TEMPLATE}

TASK
----
Generate {N} independent MDP transitions.

Each transition describes how the system evolves after an action.

STATE VALIDITY
--------------
A valid state must assign values to ALL state variables defined in the template.

Each variable must respect its domain:

- enum variables must use one of the listed values
- numeric variables must remain within their min/max range

Both "state" and "next_state" must be valid states.

Never invent new state variables.
Never omit existing state variables.

TRANSITION TYPES
----------------
Generate a mixture of VALID and INVALID transitions.

VALID transitions represent correct operation of the instrument.
They should follow realistic usage of the system and produce
a plausible next_state after the action.

INVALID transitions represent incorrect usage of the instrument.
They violate the correct operation logic of the system.

Examples of INVALID transitions may include:
- executing an action while the system is in an inappropriate state
- producing a next_state that does not correspond to the expected effect of the action
- changing state variables that should normally remain unchanged

Even INVALID transitions must still respect the state variable domains
defined in the template.

TRANSITION GENERATION
---------------------
For each transition follow two steps.

STEP 1 — GENERATE

Create a transition consisting of:

- state
- action
- next_state

state must be a valid assignment of all state variables.

action must be one of the actions defined in the template,
with all required parameters instantiated using valid values.
If the action has parameters, instantiate them with plausible values.

next_state should represent the state after the action.

Use intuition about how the system might behave.

Transitions should normally modify only the variables that are plausibly
affected by the action. Unrelated variables should usually remain unchanged.

STEP 2 — JUDGE

Evaluate the generated transition.

Assign the reward based on the transition type:

reward = 1 for VALID transitions
reward = 0 for INVALID transitions

An action is considered invalid if it does not include all required parameters.

Imperfect reasoning is allowed.
Mistakes should resemble realistic incorrect usage rather than random noise.

DATASET DIVERSITY
-----------------
Generate diverse states and actions.
Avoid repeating the same state combinations.
Explore different combinations of state variables.

ACTION FORMAT
-------------
Actions must include all their parameters explicitly.

Use the following canonical format:

action_name(parameter1=value1, parameter2=value2, ...)

The parameter names must match those defined in the template.

The parameter values must be valid values according to the template.

If an action has no parameters, use empty parentheses.

All actions must strictly follow this format.
Do not omit parameters.
Do not change parameter names.
Do not introduce new parameters.

All actions in the dataset must follow exactly the same formatting convention.

OUTPUT FORMAT
-------------
Return ONLY a valid JSON object.

The JSON object must have the following structure:

{
  "transitions": [
    {
      "state": {...},
      "action": "...",
      "next_state": {...},
      "reward": 0 or 1
    }
  ]
}

Do not include explanations.
Do not include comments.
Do not include markdown code fences.
Do not include any text outside the JSON object.
"""


def load_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_transitions(mdp_template, N, model="gpt-4o"):

    batch_size = 20
    all_transitions = []
    valid_actions = build_valid_actions(mdp_template)
    print(f"Valid actions ({len(valid_actions)}): {sorted(valid_actions)}")

    while len(all_transitions) < N:

        remaining = N - len(all_transitions)
        current_batch = min(batch_size, remaining)

        user_prompt = USER_PROMPT \
            .replace("{MDP_TEMPLATE}", json.dumps(mdp_template, indent=2)) \
            .replace("{N}", str(current_batch))

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            user=str(uuid.uuid4())
        )

        data = json.loads(response.choices[0].message.content)

        batch = data.get("transitions", [])

        # Validate each transition — discard any with hallucinated actions or invalid states
        valid_batch = []
        discarded = 0
        for t in batch:
            ok, reason = validate_transition(t, valid_actions, mdp_template)
            if ok:
                valid_batch.append(t)
            else:
                discarded += 1
                print(f"  ⚠ Discarded invalid transition: {reason} | action={t.get('action')}")

        all_transitions.extend(valid_batch)

        print(f"Generated {len(batch)} transitions, kept {len(valid_batch)}, discarded {discarded} (total={len(all_transitions)})")

    return all_transitions[:N]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("template", help="Path to MDP template JSON")
    parser.add_argument("-n", "--num", type=int, default=50,
                        help="Number of transitions to generate")
    parser.add_argument("-o", "--output", default="transitions.jsonl",
                        help="Output JSONL file")

    args = parser.parse_args()

    mdp_template = load_template(args.template)

    transitions = generate_transitions(mdp_template, args.num)

    with open(args.output, "w", encoding="utf-8") as f:
        for t in transitions:
            f.write(json.dumps(t) + "\n")
    
    print(f"Saved {len(transitions)} transitions to {args.output}")


if __name__ == "__main__":
    main()
