"""Exhaustively generate structurally valid MDP candidates, then judge them."""

import argparse
import json
import random
import re
from collections import Counter, defaultdict, deque
from itertools import product

from mdp_generator import DEFAULT_MODELS, call_api, make_client


JUDGE_SYSTEM_PROMPT = """
You judge fixed Markov Decision Process transitions.

For each supplied candidate, evaluate the exact relationship between its initial `state`, its `action`, and its supplied `next_state`.

Judge the candidate exactly as supplied. Do not rewrite, repair, complete, add, remove, or replace any part of the transition.

Assign reward `1` only when the complete transition correctly represents successful execution of the action.

Successful execution requires all applicable conditions below:

* every required precondition is satisfied in the initial state;
* every expected postcondition is satisfied in the next state;
* every required effect is represented completely;
* effects that depend on one another are mutually consistent;
* no unrelated state variable changes;
* no supplied state change contradicts the action;
* all state values are permitted by the template;
* the transition is internally consistent as a whole.

Do not judge success solely from whether the state changed.

A transition may be successful even when no represented value changes, provided that the action is valid in the initial state and the supplied next state correctly satisfies all expected postconditions.

A transition may be unsuccessful even when represented values change, if the required preconditions or postconditions are not fully satisfied.

Assign reward `0` when the supplied transition does not establish complete successful execution.

This includes cases where:

* a required precondition is absent;
* an expected postcondition is missing or incorrect;
* a required effect is incomplete;
* dependent effects are inconsistent;
* an unrelated or unsupported state changes;
* the transition contradicts the action;
* the available information is insufficient to confirm successful execution.

Reward `0` does not necessarily mean that the transition is physically impossible. It means that the complete supplied transition does not satisfy the criteria for successful execution.

Use the supplied template as the source of truth for:

* valid state variables and values;
* action semantics;
* involved objects and components;
* represented preconditions, postconditions, and effects.

Do not invent unsupported domain-specific rules.

Do not confuse a value expected after the action with a value required before the action.

Evaluate the whole transition before assigning the reward.

Return only the structured judgments requested in the user message, with no explanation, commentary, markdown, or additional fields.
"""

JUDGE_USER_PROMPT = """
MDP TEMPLATE
------------
{template}

CANDIDATES
----------
{candidates}

Return only this JSON shape:
{{"judgments":[{{"id":0,"reward":0}}]}}

Use each supplied ID exactly once. Reward must be 0 or 1.
"""


PERSISTENT_COMPONENT_TYPES = {
    "binary control",
    "continuous control",
    "selector control",
    "actuator",
}
MOVEMENT_SUBTYPES = {
    "place",
    "move",
    "insert",
    "remove",
    "retrieve",
    "load",
    "unload",
}
EMPTY_MATERIAL_VALUES = ("material:none", "none", "empty")
EMPTY_RECEPTOR_VALUES = ("empty", "none", None)


def load_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def object_base(label):
    if not isinstance(label, str):
        return ""
    return re.sub(r"\[\d+\]$", "", label)


def object_matches(left, right):
    return bool(left and right) and object_base(left) == object_base(right)


def state_owner(state, template_object):
    if state.get("owner"):
        return state["owner"]
    name = state["name"]
    prefix = name.rsplit(".", 1)[0]
    return prefix if "." in name and ":" in prefix else template_object


def state_local_name(state):
    return state["name"].rsplit(".", 1)[-1]


def infer_state_roles(template):
    """Return descriptors with internal roles, retaining legacy inference."""
    controlled_components = {
        action["name"].rsplit(".", 1)[0].rsplit(".", 1)[-1]
        for action in template["actions"]
        if action.get("type") == "control" and "." in action["name"]
    }
    receptor_names = set()
    for action in template["actions"]:
        if action.get("subtype") not in MOVEMENT_SUBTYPES:
            continue
        params = action.get("parameters", {})
        receptor = (
            action.get("destination_receptor")
            or action.get("target_receptor")
            or params.get("receptor")
        )
        if receptor:
            receptor_names.add(receptor)
        target = params.get("target")
        if isinstance(target, str) and ":" not in target:
            receptor_names.add(target)

    descriptors = []
    for original in template["states"]:
        state = dict(original)
        local = state_local_name(state)
        component_type = state.get("component_type")
        state_type = state.get("state_type")
        if state_type in ("material", "location"):
            role = state_type
        elif component_type == "momentary control":
            role = "momentary"
        elif component_type in PERSISTENT_COMPONENT_TYPES:
            role = "control"
        elif component_type in ("observable", "receptor"):
            role = component_type
        else:
            role = state.get("role")

        if not role:
            if local == "material":
                role = "material"
            elif local == "location":
                role = "location"
            elif local in receptor_names:
                role = "receptor"
            elif local in controlled_components:
                role = "control"
            else:
                role = "observable"
        state["role"] = role
        state["owner"] = state_owner(state, template["object"])
        descriptors.append(state)
    return descriptors


def action_string(action, parameters):
    if not parameters:
        return f'{action["name"]}()'
    values = ", ".join(f"{key}={value}" for key, value in parameters.items())
    return f'{action["name"]}({values})'


def instantiate_actions(template):
    """Instantiate every template action parameter product deterministically."""
    result = []
    for action in template["actions"]:
        params = action.get("parameters", {})
        keys = list(params)
        domains = [value if isinstance(value, list) else [value] for value in params.values()]
        combinations = product(*domains) if domains else [()]
        for values in combinations:
            instance = dict(action)
            instance["instantiated_parameters"] = dict(zip(keys, values))
            instance["canonical"] = action_string(
                instance, instance["instantiated_parameters"]
            )
            result.append(instance)
    return result


def state_domain(state):
    if state["kind"] == "enum":
        return list(state["values"])
    minimum = state.get("min", 0)
    maximum = state.get("max", minimum)
    step = state.get("step")
    if step and step > 0:
        values = []
        value = minimum
        while value <= maximum:
            values.append(value)
            value += step
        if not values or values[-1] != maximum:
            values.append(maximum)
        return list(dict.fromkeys(values))
    midpoint = (minimum + maximum) / 2
    if isinstance(minimum, int) and isinstance(maximum, int):
        midpoint = int(round(midpoint))
    return list(dict.fromkeys([minimum, midpoint, maximum]))


def cartesian_state_rows(states):
    """Enumerate the full Cartesian product of all represented state domains."""
    names = [state["name"] for state in states]
    domains = [state_domain(state) for state in states]
    if any(not domain for domain in domains):
        return []
    return [dict(zip(names, values)) for values in product(*domains)]


def _location_points_to_receptor(location, receptor):
    if not isinstance(location, str):
        return False
    local = state_local_name(receptor)
    suffix = f":{local}"
    if not location.endswith(suffix):
        return False
    return object_matches(location[:-len(suffix)], receptor["owner"])


def is_valid_composite_state(state, template):
    """Reject only explicit receptor/location contradictions."""
    states = infer_state_roles(template)
    locations = [item for item in states if item["role"] == "location"]
    receptors = [item for item in states if item["role"] == "receptor"]
    for location_state in locations:
        movable = location_state["owner"]
        location = state.get(location_state["name"])
        for receptor in receptors:
            occupancy = state.get(receptor["name"])
            location_claims_receptor = _location_points_to_receptor(location, receptor)
            receptor_claims_movable = object_matches(occupancy, movable)
            if location_claims_receptor != receptor_claims_movable:
                return False
    return True


def observable_states_for_owners(states, owners):
    """Return all observable descriptors owned by directly involved objects."""
    return [
        state for state in states
        if state["role"] == "observable"
        and any(object_matches(state["owner"], owner) for owner in owners if owner)
    ]


def _state_for_component(states, owner, component):
    return next(
        (
            state for state in states
            if state_local_name(state) == component
            and object_matches(state["owner"], owner)
        ),
        None,
    )


def _action_component_type(action, states, template_object):
    if action.get("component_type"):
        return action["component_type"]
    owner = action.get("owner", template_object)
    component = action.get("component")
    if component:
        descriptor = _state_for_component(states, owner, component)
        if descriptor:
            return descriptor.get("component_type")
        # Legacy templates omitted action component_type. A control action with
        # no represented component state and no requested value was how
        # momentary controls such as tare_button.press() were encoded.
        if "value" not in action.get("instantiated_parameters", action.get("parameters", {})):
            return "momentary control"
    return None


def action_category(action, template, states=None):
    states = states or infer_state_roles(template)
    if action.get("type") == "interaction":
        subtype = str(action.get("subtype", "")).lower()
        if subtype == "transfer":
            return "material transfer"
        if subtype in MOVEMENT_SUBTYPES:
            return "movement"
        return "other"
    if action.get("type") == "control":
        component_type = _action_component_type(action, states, template["object"])
        if component_type == "momentary control":
            return "momentary control"
        return "persistent control"
    return "other"


def _observable_products(state, observables, primary_effect):
    """Merge a fixed primary effect with the full observable value product."""
    observables = [item for item in observables if item["name"] not in primary_effect]
    if not observables:
        return [dict(primary_effect)]
    names = [item["name"] for item in observables]
    domains = [state_domain(item) for item in observables]
    effects = []
    for values in product(*domains):
        effect = dict(primary_effect)
        effect.update(zip(names, values))
        effects.append(effect)
    return effects


def generate_noop_effect(state, action, template):
    return {}


def generate_persistent_control_effects(state, action, template, states=None):
    states = states or infer_state_roles(template)
    params = action["instantiated_parameters"]
    owner = action.get("owner", template["object"])
    component = action.get("component") or action["name"].rsplit(".", 1)[0].rsplit(".", 1)[-1]
    associated = _state_for_component(states, owner, component)
    if not associated:
        return []

    requested_values = []
    if "value" in params:
        if params["value"] in state_domain(associated):
            requested_values = [params["value"]]
    else:
        requested_values = state_domain(associated)

    observables = observable_states_for_owners(states, [owner])
    effects = []
    for requested in requested_values:
        # A successful persistent action must change its associated component.
        if state.get(associated["name"]) == requested:
            continue
        effects.extend(
            _observable_products(state, observables, {associated["name"]: requested})
        )
    return effects


def generate_momentary_control_effects(state, action, template, states=None):
    states = states or infer_state_roles(template)
    owner = action.get("owner", template["object"])
    mutable = [item for item in states if object_matches(item["owner"], owner)]
    names = [item["name"] for item in mutable]
    domains = [state_domain(item) for item in mutable]
    effects = []
    for values in product(*domains):
        effect = dict(zip(names, values))
        if any(state.get(name) != value for name, value in effect.items()):
            effects.append(effect)
    return effects


def _material_states_for_owner(states, owner):
    return [
        item for item in states
        if item["role"] == "material" and object_matches(item["owner"], owner)
    ]


def _empty_value(state_descriptor, candidates):
    return next(
        (value for value in state_domain(state_descriptor) if value in candidates),
        None,
    )


def _interaction_owners(action, template):
    params = action["instantiated_parameters"]
    return list(dict.fromkeys(
        owner for owner in (
            action.get("source_object") or params.get("source_object"),
            action.get("target_object") or params.get("target_object"),
            action.get("movable_object") or params.get("movable_object") or params.get("object"),
            action.get("destination_object"),
        ) if owner
    ))


def generate_transfer_effects(state, action, template, states=None):
    states = states or infer_state_roles(template)
    params = action["instantiated_parameters"]
    source = action.get("source_object") or params.get("source_object")
    target = action.get("target_object") or params.get("target_object")
    material = action.get("material") or params.get("material")
    primary = {}

    for descriptor in _material_states_for_owner(states, source):
        empty = _empty_value(descriptor, EMPTY_MATERIAL_VALUES)
        if empty is not None:
            primary[descriptor["name"]] = empty
    for descriptor in _material_states_for_owner(states, target):
        if material in state_domain(descriptor):
            primary[descriptor["name"]] = material

    if not primary:
        return []
    observables = observable_states_for_owners(
        states, _interaction_owners(action, template)
    )
    return _observable_products(state, observables, primary)


def _normalize_receptor_reference(reference, default_owner=None):
    if isinstance(reference, dict):
        return reference.get("owner") or default_owner, reference.get("component") or reference.get("name")
    if not isinstance(reference, str):
        return default_owner, None
    if "." in reference and ":" in reference.rsplit(".", 1)[0]:
        return reference.rsplit(".", 1)
    return default_owner, reference


def _movement_metadata(action):
    params = action["instantiated_parameters"]
    subtype = str(action.get("subtype", "")).lower()
    movable = (
        action.get("movable_object")
        or params.get("movable_object")
        or params.get("object")
        or action.get("source_object")
    )
    source_owner = action.get("source_receptor_owner")
    source_owner, source_receptor = _normalize_receptor_reference(
        action.get("source_receptor") or params.get("source_receptor"), source_owner
    )
    destination_owner = (
        action.get("destination_object")
        or action.get("target_object")
        or params.get("target_object")
    )
    destination_ref = (
        action.get("destination_receptor")
        or action.get("target_receptor")
        or params.get("destination_receptor")
        or params.get("receptor")
    )
    raw_target = params.get("target")
    if not destination_ref and isinstance(raw_target, str) and ":" not in raw_target:
        destination_ref = raw_target
    if not destination_owner and isinstance(raw_target, str) and ":" in raw_target:
        destination_owner = raw_target
    destination_owner, destination_receptor = _normalize_receptor_reference(
        destination_ref, destination_owner
    )
    destination_location = action.get("destination_location") or params.get("destination_location")
    if not destination_location and subtype in ("remove", "retrieve", "unload"):
        destination_location = params.get("target_location")
    return {
        "movable": movable,
        "source_owner": source_owner,
        "source_receptor": source_receptor,
        "destination_owner": destination_owner,
        "destination_receptor": destination_receptor,
        "destination_location": destination_location,
        "uses_legacy_movable_inference": not bool(
            action.get("movable_object") or params.get("movable_object")
        ),
    }


def _receptor_state(states, owner, receptor_name):
    if not receptor_name:
        return None
    candidates = [
        item for item in states
        if item["role"] == "receptor" and state_local_name(item) == receptor_name
    ]
    if owner:
        candidates = [item for item in candidates if object_matches(item["owner"], owner)]
    return candidates[0] if len(candidates) == 1 else None


def _receptor_occupancy_value(descriptor, movable):
    return next(
        (value for value in state_domain(descriptor) if object_matches(value, movable)),
        None,
    )


def _location_value_for_destination(descriptor, metadata):
    domain = state_domain(descriptor)
    if metadata["destination_location"] in domain:
        return metadata["destination_location"]
    receptor = metadata["destination_receptor"]
    owner = metadata["destination_owner"]
    if receptor:
        return next(
            (
                value for value in domain
                if isinstance(value, str)
                and value.endswith(f":{receptor}")
                and (not owner or object_matches(value[:-(len(receptor) + 1)], owner))
            ),
            None,
        )
    if owner:
        return next((value for value in domain if object_matches(value, owner)), None)
    return None


def _movement_primary_effect(action, template, states):
    metadata = _movement_metadata(action)
    movable = metadata["movable"]
    if not movable:
        return {}, metadata
    primary = {}

    source = _receptor_state(
        states, metadata["source_owner"], metadata["source_receptor"]
    )
    if source:
        empty = _empty_value(source, EMPTY_RECEPTOR_VALUES)
        if empty is not None:
            primary[source["name"]] = empty

    destination = _receptor_state(
        states, metadata["destination_owner"], metadata["destination_receptor"]
    )
    if destination:
        occupancy = _receptor_occupancy_value(destination, movable)
        if occupancy is not None:
            primary[destination["name"]] = occupancy

    for descriptor in states:
        if descriptor["role"] != "location" or not object_matches(descriptor["owner"], movable):
            continue
        location = _location_value_for_destination(descriptor, metadata)
        if location is not None:
            primary[descriptor["name"]] = location
    return primary, metadata


def generate_movement_effects(state, action, template, states=None):
    states = states or infer_state_roles(template)
    primary, metadata = _movement_primary_effect(action, template, states)
    if not primary:
        return []
    owners = _interaction_owners(action, template)
    owners.extend(
        owner for owner in (
            metadata["movable"], metadata["source_owner"], metadata["destination_owner"]
        ) if owner
    )
    observables = observable_states_for_owners(states, list(dict.fromkeys(owners)))
    return _observable_products(state, observables, primary)


def _effect_scope(action, template, states):
    category = action_category(action, template, states)
    if category == "persistent control":
        owner = action.get("owner", template["object"])
        component = action.get("component") or action["name"].rsplit(".", 1)[0].rsplit(".", 1)[-1]
        associated = _state_for_component(states, owner, component)
        names = {associated["name"]} if associated else set()
        names.update(item["name"] for item in observable_states_for_owners(states, [owner]))
        return names
    if category == "momentary control":
        owner = action.get("owner", template["object"])
        return {item["name"] for item in states if object_matches(item["owner"], owner)}
    if category == "material transfer":
        params = action["instantiated_parameters"]
        source = action.get("source_object") or params.get("source_object")
        target = action.get("target_object") or params.get("target_object")
        names = {
            item["name"] for item in states
            if item["role"] == "material"
            and (object_matches(item["owner"], source) or object_matches(item["owner"], target))
        }
        names.update(
            item["name"] for item in observable_states_for_owners(
                states, _interaction_owners(action, template)
            )
        )
        return names
    if category == "movement":
        primary, metadata = _movement_primary_effect(action, template, states)
        owners = _interaction_owners(action, template)
        owners.extend(
            owner for owner in (
                metadata["movable"], metadata["source_owner"], metadata["destination_owner"]
            ) if owner
        )
        names = set(primary)
        names.update(
            item["name"] for item in observable_states_for_owners(
                states, list(dict.fromkeys(owners))
            )
        )
        return names
    return set()


def permitted_changed_variables(action, template):
    return _effect_scope(action, template, infer_state_roles(template))


def _expected_transfer_primary(action, template, states):
    effects = generate_transfer_effects({}, action, template, states)
    if not effects:
        return {}
    observable_names = {
        item["name"] for item in observable_states_for_owners(
            states, _interaction_owners(action, template)
        )
    }
    return {key: value for key, value in effects[0].items() if key not in observable_names}


def is_structurally_valid_transition(state, action, next_state, template):
    states = infer_state_roles(template)
    names = {item["name"] for item in states}
    if set(state) != names or set(next_state) != names:
        return False
    by_name = {item["name"]: item for item in states}
    for values in (state, next_state):
        for name, value in values.items():
            if value not in state_domain(by_name[name]):
                return False
    if not is_valid_composite_state(state, template) or not is_valid_composite_state(next_state, template):
        return False

    changed = {name for name in state if state[name] != next_state[name]}
    if not changed:
        return True
    if not changed.issubset(_effect_scope(action, template, states)):
        return False

    category = action_category(action, template, states)
    if category == "persistent control":
        owner = action.get("owner", template["object"])
        component = action.get("component") or action["name"].rsplit(".", 1)[0].rsplit(".", 1)[-1]
        associated = _state_for_component(states, owner, component)
        params = action["instantiated_parameters"]
        if not associated or associated["name"] not in changed:
            return False
        if "value" in params and next_state[associated["name"]] != params["value"]:
            return False
    elif category == "material transfer":
        expected = _expected_transfer_primary(action, template, states)
        if not expected or any(next_state.get(name) != value for name, value in expected.items()):
            return False
    elif category == "movement":
        expected, _ = _movement_primary_effect(action, template, states)
        if not expected or any(next_state.get(name) != value for name, value in expected.items()):
            return False
    elif category == "other":
        return False
    return True


def _effects_for_action(state, action, template, states):
    category = action_category(action, template, states)
    if category == "persistent control":
        return generate_persistent_control_effects(state, action, template, states)
    if category == "momentary control":
        return generate_momentary_control_effects(state, action, template, states)
    if category == "material transfer":
        return generate_transfer_effects(state, action, template, states)
    if category == "movement":
        return generate_movement_effects(state, action, template, states)
    return []


def _candidate_key(candidate):
    return (
        tuple(sorted(candidate["state"].items())),
        candidate["action"],
        tuple(sorted(candidate["next_state"].items())),
    )


def _balanced_sample(candidates, limit, seed):
    if limit is None or limit >= len(candidates):
        return list(candidates)
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate["action"]].append(candidate)
    rng = random.Random(seed)
    queues = []
    for action in sorted(groups):
        group = list(groups[action])
        rng.shuffle(group)
        queues.append(deque(group))
    selected = []
    while len(selected) < limit and any(queues):
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.popleft())
    return selected


def generate_candidates(
    template, limit=None, seed=0, return_metrics=False, include_noops=True
):
    """Build the exhaustive candidate set, then optionally sample it."""
    states = infer_state_roles(template)
    actions = instantiate_actions(template)
    raw_rows = cartesian_state_rows(states)
    valid_rows = [row for row in raw_rows if is_valid_composite_state(row, template)]

    metrics = {
        "raw_cartesian_state_count": len(raw_rows),
        "invalid_composite_state_count": len(raw_rows) - len(valid_rows),
        "valid_initial_state_count": len(valid_rows),
        "instantiated_action_count": len(actions),
        "state_action_pair_count": len(valid_rows) * len(actions),
        "candidate_count_before_structural_filter": 0,
        "structurally_rejected_count": 0,
        "candidate_count_before_deduplication": 0,
        "exact_duplicates_removed": 0,
        "full_candidate_count": 0,
        "sampled_candidate_count": 0,
        "sampling_seed": seed if limit is not None else None,
        "sample_limit": limit,
        "per_action_counts_full": {},
        "per_action_counts_sampled": {},
        "per_category_counts_full": {},
        "per_category_counts_sampled": {},
        "legacy_movement_metadata_actions": [],
    }
    if not states or not actions:
        return ([], metrics) if return_metrics else []

    action_by_canonical = {action["canonical"]: action for action in actions}
    for action in actions:
        if action_category(action, template, states) == "movement":
            _, metadata = _movement_primary_effect(action, template, states)
            if metadata["uses_legacy_movable_inference"]:
                metrics["legacy_movement_metadata_actions"].append(action["canonical"])

    candidates = []
    seen = set()
    for state in valid_rows:
        for action in actions:
            effects = [generate_noop_effect(state, action, template)]
            effects.extend(_effects_for_action(state, action, template, states))
            for effect in effects:
                metrics["candidate_count_before_structural_filter"] += 1
                next_state = dict(state)
                next_state.update(effect)
                candidate = {
                    "state": dict(state),
                    "action": action["canonical"],
                    "next_state": next_state,
                }
                if not is_structurally_valid_transition(state, action, next_state, template):
                    metrics["structurally_rejected_count"] += 1
                    continue
                if not include_noops and next_state == state:
                    continue
                metrics["candidate_count_before_deduplication"] += 1
                key = _candidate_key(candidate)
                if key in seen:
                    metrics["exact_duplicates_removed"] += 1
                    continue
                seen.add(key)
                candidates.append(candidate)

    full_action_counts = Counter(candidate["action"] for candidate in candidates)
    full_category_counts = Counter(
        action_category(action_by_canonical[candidate["action"]], template, states)
        for candidate in candidates
    )
    metrics["full_candidate_count"] = len(candidates)
    metrics["per_action_counts_full"] = dict(full_action_counts)
    metrics["per_category_counts_full"] = dict(full_category_counts)

    sampled = _balanced_sample(candidates, limit, seed)
    sampled_action_counts = Counter(candidate["action"] for candidate in sampled)
    sampled_category_counts = Counter(
        action_category(action_by_canonical[candidate["action"]], template, states)
        for candidate in sampled
    )
    metrics["sampled_candidate_count"] = len(sampled)
    metrics["per_action_counts_sampled"] = dict(sampled_action_counts)
    metrics["per_category_counts_sampled"] = dict(sampled_category_counts)
    metrics["noop_count_full"] = sum(
        candidate["state"] == candidate["next_state"] for candidate in candidates
    )
    metrics["noop_percentage_full"] = round(
        100 * metrics["noop_count_full"] / len(candidates), 2
    ) if candidates else 0.0
    metrics["noop_count_sampled"] = sum(
        candidate["state"] == candidate["next_state"] for candidate in sampled
    )
    metrics["noop_percentage_sampled"] = round(
        100 * metrics["noop_count_sampled"] / len(sampled), 2
    ) if sampled else 0.0
    return (sampled, metrics) if return_metrics else sampled


def print_generation_metrics(metrics):
    print(f"Raw Cartesian initial states: {metrics['raw_cartesian_state_count']}")
    print(f"Invalid composite states filtered: {metrics['invalid_composite_state_count']}")
    print(f"Valid initial states: {metrics['valid_initial_state_count']}")
    print(f"Instantiated actions: {metrics['instantiated_action_count']}")
    print(f"State-action pairs: {metrics['state_action_pair_count']}")
    print(
        "Candidates before structural filtering: "
        f"{metrics['candidate_count_before_structural_filter']}"
    )
    print(f"Candidates rejected structurally: {metrics['structurally_rejected_count']}")
    print(
        "Candidates before deduplication: "
        f"{metrics['candidate_count_before_deduplication']}"
    )
    print(f"Exact duplicates removed: {metrics['exact_duplicates_removed']}")
    print(f"Final exhaustive candidates: {metrics['full_candidate_count']}")
    print(
        f"No-ops: {metrics['noop_count_full']} "
        f"({metrics['noop_percentage_full']:.2f}%)"
    )
    print("Candidate counts per action:")
    for action, count in metrics["per_action_counts_full"].items():
        print(f"  {action}: {count}")
    print("Candidate counts by action category:")
    for category, count in metrics["per_category_counts_full"].items():
        print(f"  {category}: {count}")
    if metrics["sample_limit"] is not None:
        print(
            f"Post-generation sample: {metrics['sampled_candidate_count']} "
            f"of {metrics['full_candidate_count']} (seed={metrics['sampling_seed']})"
        )
        print("Sampled candidate counts per action:")
        for action, count in metrics["per_action_counts_sampled"].items():
            print(f"  {action}: {count}")
        print("Sampled candidate counts by action category:")
        for category, count in metrics["per_category_counts_sampled"].items():
            print(f"  {category}: {count}")
    if metrics["legacy_movement_metadata_actions"]:
        print("WARNING: movement actions using ambiguous legacy movable-object inference:")
        for action in metrics["legacy_movement_metadata_actions"]:
            print(f"  {action}")


def parse_judgments(raw, expected_ids):
    judgments = json.loads(raw).get("judgments", [])
    result = {}
    for judgment in judgments:
        candidate_id = judgment.get("id")
        reward = judgment.get("reward")
        if candidate_id in expected_ids and reward in (0, 1) and candidate_id not in result:
            result[candidate_id] = int(reward)
    missing = expected_ids - set(result)
    if missing:
        raise ValueError(f"missing or invalid judgments for IDs: {sorted(missing)}")
    return result


def judge_candidates(template, candidates, provider, model, batch_size=50):
    client = make_client(provider)
    judged = []
    print(f"Judge provider: {provider}  Model: {model}  Batch size: {batch_size}")
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        payload = [dict(candidate, id=index) for index, candidate in enumerate(batch)]
        prompt = JUDGE_USER_PROMPT.format(
            template=json.dumps(template, indent=2),
            candidates=json.dumps(payload, indent=2),
        )
        expected_ids = set(range(len(batch)))
        last_error = None
        for attempt in range(3):
            try:
                raw = call_api(
                    client, provider, model, JUDGE_SYSTEM_PROMPT, prompt,
                    max_tokens=4096, temperature=0,
                )
                rewards = parse_judgments(raw, expected_ids)
                break
            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
                print(f"  Invalid judgment response (attempt {attempt + 1}/3): {error}")
        else:
            raise RuntimeError(f"LLM failed to judge batch starting at {start}: {last_error}")

        for index, candidate in enumerate(batch):
            judged.append({**candidate, "reward": rewards[index]})
        print(f"Judged {len(judged)}/{len(candidates)} candidates")
    return judged


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustively generate structural MDP candidates and judge them"
    )
    parser.add_argument("template", help="Path to MDP template JSON")
    parser.add_argument(
        "-n", "--num", "--sample-size", dest="sample_size", type=int, default=None,
        help=(
            "Optional deterministic post-generation sample size. The complete "
            "candidate set is always constructed first."
        ),
    )
    parser.add_argument("-o", "--output", default="transitions.jsonl",
                        help="Judged output JSONL file")
    parser.add_argument("--candidates-output", default=None,
                        help="Optional JSONL file for candidates before LLM judging")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                        help="LLM provider used only for judging")
    parser.add_argument("--model", default=None, help="Judge model name override")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed used only for optional post-generation sampling")
    parser.add_argument("--judge-batch-size", type=int, default=50,
                        help="Candidates per judge request")
    parser.add_argument(
        "--exclude-noops", action="store_true",
        help="Exclude candidates whose next state equals their initial state",
    )
    args = parser.parse_args()

    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be at least 1")
    if args.judge_batch_size < 1:
        parser.error("--judge-batch-size must be at least 1")

    template = load_template(args.template)
    candidates, metrics = generate_candidates(
        template, args.sample_size, args.seed, return_metrics=True,
        include_noops=not args.exclude_noops,
    )
    print_generation_metrics(metrics)
    if not candidates:
        raise SystemExit("No candidates could be generated from this template")

    if args.candidates_output:
        with open(args.candidates_output, "w", encoding="utf-8") as f:
            for candidate in candidates:
                f.write(json.dumps(candidate) + "\n")
        print(f"Saved {len(candidates)} unjudged candidates to {args.candidates_output}")

    model = args.model or DEFAULT_MODELS[args.provider]
    transitions = judge_candidates(
        template, candidates, args.provider, model, args.judge_batch_size
    )
    with open(args.output, "w", encoding="utf-8") as f:
        for transition in transitions:
            f.write(json.dumps(transition) + "\n")
    reward_counts = Counter(item["reward"] for item in transitions)
    noop_counts = Counter(
        item["reward"] for item in transitions
        if item["state"] == item["next_state"]
    )
    print("Reward policy: binary")
    for reward in (1, 0):
        count = reward_counts[reward]
        percentage = 100 * count / len(transitions) if transitions else 0
        print(f"Reward {reward}: {count} ({percentage:.2f}%)")
    print(f"No-op counts by reward: {dict(sorted(noop_counts.items(), reverse=True))}")
    print(f"Saved {len(transitions)} judged transitions to {args.output}")


if __name__ == "__main__":
    main()
