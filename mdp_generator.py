import uuid
import json
import argparse
from itertools import product

def make_client(provider: str):
    """Return an API client for the given provider ('openai' or 'anthropic')."""
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    else:
        from openai import OpenAI
        return OpenAI()


def call_api(client, provider: str, model: str, system: str, user: str,
             max_tokens: int = 16000, temperature: float = 0.7) -> str:
    """Call the appropriate API and return the raw response text."""
    if provider == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=1,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": user,
                     "cache_control": {"type": "ephemeral"}}
                    ]
                 }
            ]
        )
        return response.content[0].text
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=temperature,
            user=str(uuid.uuid4())
        )
        return response.choices[0].message.content


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

def normalize_action(action: str) -> str:
    """Canonical form: strip spaces around '=', sort params."""
    import re
    if not isinstance(action, str):
        return action
    action = action.strip()
    m = re.match(r"^(.*?)\((.*)\)$", action)
    if not m:
        return action.replace(" ", "")
    name, params = m.group(1), m.group(2)
    if not params.strip():
        return f"{name}()"
    parts = []
    for p in params.split(","):
        p = p.strip()
        if not p:
            continue
        if "=" in p:
            k, v = p.split("=", 1)
            p = f"{k.strip()}={v.strip()}"
        parts.append(p)
    return f"{name}({', '.join(sorted(parts))})"


def repair_action(action: str, valid_actions: set) -> str:
    """Map a malformed action to its canonical form where the intent is unambiguous.

    Handles:
      1. Missing key prefix: 'f(material:cuso4)' -> key was omitted entirely
      2. Missing namespace prefix on value: 'f(material=cuso4)' where valid is
         'f(material=material:cuso4)' — the category prefix was dropped from the value.

    Does NOT repair missing parameters (ambiguous intent).
    Returns the canonical action if repair is unambiguous, else the original.
    """
    import re

    if action in valid_actions:
        return action

    m = re.match(r"^(.*?)\((.*)\)$", action.strip())
    if not m:
        return action

    name = m.group(1).strip()
    raw_params = m.group(2).strip()

    # Must have parameters — we don't repair missing-param calls
    if not raw_params:
        return action

    # Collect candidates with the same action name
    candidates = [a for a in valid_actions if a.startswith(f"{name}(")]
    if not candidates:
        return action

    # Parse raw parameter values, handling both '=' and ':' as separators.
    # gpt-4o-mini sometimes uses 'key:value' instead of 'key=value'.
    # When ':' is the separator, strip the key part (left of first ':').
    raw_vals = []
    for p in raw_params.split(","):
        p = p.strip()
        if not p:
            continue
        if "=" in p:
            raw_vals.append(p.split("=", 1)[1].strip())
        elif ":" in p:
            # 'key:value' format — strip the key, keep the value
            raw_vals.append(p.split(":", 1)[1].strip())
        else:
            raw_vals.append(p)

    def values_match(raw_v, canonical_v):
        """True if raw_v matches canonical_v exactly or is a suffix after ':'."""
        return raw_v == canonical_v or canonical_v.endswith(f":{raw_v}")

    for candidate in candidates:
        cm = re.match(r"^(.*?)\((.*)\)$", candidate)
        if not cm:
            continue
        cparams = cm.group(2)
        cvals = []
        for p in cparams.split(","):
            p = p.strip()
            cvals.append(p.split("=", 1)[1].strip() if "=" in p else p)

        if len(raw_vals) != len(cvals):
            continue
        if all(values_match(r, c) for r, c in zip(raw_vals, cvals)):
            return candidate

    return action


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
    raw = t.get("action", "")
    canonical = normalize_action(raw)
    if canonical not in valid_actions:
        canonical = repair_action(raw, valid_actions)
    if canonical not in valid_actions:
        return False, f"unknown action: {raw!r}"
    t["action"] = canonical
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

USER_PROMPT_DELTA = USER_PROMPT.replace(
    '      "next_state": {...},',
    '      "changes": {...},'
).replace(
    'Both "state" and "next_state" must be valid states.',
    'Both "state" and the reconstructed next_state must assign valid values.'
).replace(
    'Do not include any text outside the JSON object.',
    'Do not include any text outside the JSON object.'
  # "changes" instruction appended below
) + """

CHANGES FORMAT
--------------
"changes" must contain ONLY the state variables whose value differs between
state and next_state. Omit variables that do not change.
If nothing changes, use an empty object: "changes": {}.
All values in "changes" must be valid for their respective variable.
"""


def load_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_transitions(mdp_template, N, provider="openai", model="gpt-4o",
                        delta=False):
    """Generate N MDP transitions.

    Args:
        provider:  'openai' or 'anthropic'
        model:     model name for the chosen provider
        delta:     if True, ask the LLM to output only changed variables
                   ('changes' field) instead of the full next_state dict.
                   next_state is reconstructed locally before saving.
    """
    batch_size = 100
    all_transitions = []
    valid_actions = build_valid_actions(mdp_template)
    client = make_client(provider)
    print(f"Provider: {provider}  Model: {model}  Delta: {delta}")
    print(f"Valid actions ({len(valid_actions)}): {sorted(valid_actions)}")

    prompt = USER_PROMPT_DELTA if delta else USER_PROMPT

    while len(all_transitions) < N:

        remaining = N - len(all_transitions)
        current_batch = min(batch_size, remaining)

        user_prompt = prompt \
            .replace("{MDP_TEMPLATE}", json.dumps(mdp_template, indent=2)) \
            .replace("{N}", str(current_batch))

        try:
            raw = call_api(client, provider, model, SYSTEM_PROMPT, user_prompt)
            data = json.loads(raw)
        except json.JSONDecodeError:
            batch_size = max(10, batch_size // 2)
            print(f"  ⚠ JSON truncated — reducing batch size to {batch_size} and retrying")
            continue

        batch = data.get("transitions", [])

        valid_batch = []
        discarded = 0
        for t in batch:
            # Delta mode: reconstruct next_state from state + changes
            if delta:
                changes = t.pop("changes", {})
                t["next_state"] = {**t.get("state", {}), **changes}

            ok, reason = validate_transition(t, valid_actions, mdp_template)
            if ok:
                valid_batch.append(t)
            else:
                discarded += 1
                print(f"  ⚠ Discarded invalid transition: {reason} | action={t.get('action')}")

        all_transitions.extend(valid_batch)
        print(f"Generated {len(batch)} transitions, kept {len(valid_batch)}, "
              f"discarded {discarded} (total={len(all_transitions)})")

    return all_transitions[:N]


DEFAULT_MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-4-6"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("template", help="Path to MDP template JSON")
    parser.add_argument("-n", "--num", type=int, default=50,
                        help="Number of transitions to generate")
    parser.add_argument("-o", "--output", default="transitions.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--provider", choices=["openai", "anthropic"],
                        default="openai",
                        help="LLM provider (default: openai)")
    parser.add_argument("--model", default=None,
                        help="Model name. Defaults: openai=gpt-4o, "
                             "anthropic=claude-sonnet-4-6")
    parser.add_argument("--delta", action="store_true",
                        help="Ask LLM to output only changed variables "
                             "(reduces output tokens ~40%%)")

    args = parser.parse_args()

    model = args.model or DEFAULT_MODELS[args.provider]
    mdp_template = load_template(args.template)

    transitions = generate_transitions(
        mdp_template, args.num,
        provider=args.provider, model=model, delta=args.delta
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for t in transitions:
            f.write(json.dumps(t) + "\n")
    
    print(f"Saved {len(transitions)} transitions to {args.output}")


if __name__ == "__main__":
    main()
