"""Observation extraction for trusted evaluator-side deterministic lookups."""


def observation_for(lookup, hidden_state, action_id, *, observation_id):
    """Return the mapped observation with environment-owned identity and source.

    The caller is a trusted environment/evaluator. Neither the lookup object nor
    hidden_state is an agent-facing input or output. The environment assigns
    observation_id; the agent supplies only the selected action.
    """
    try:
        observation = lookup["action_results"][hidden_state][action_id]["observation"]
    except (KeyError, TypeError):
        raise ValueError("No observation is defined for the requested state/action") from None
    if not isinstance(observation, str):
        raise ValueError("Observation must be text")
    if not isinstance(observation_id, str) or not observation_id:
        raise ValueError("The environment must assign an observation ID")
    return {"observation_id": observation_id, "content": observation,
            "source": "environment"}
