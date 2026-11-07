"""Trusted assembly of agent submissions and environment feedback."""

from copy import deepcopy

from .validate import validate_agent_submission, validators


def make_round(*, round_index, before_action, environment_feedback, after_observation):
    """Combine separately owned fields into one trajectory round.

    The environment/logger supplies the index and feedback directly. Agent
    submissions cannot include an observation, its source, or remaining budget.
    """
    for phase, submission in (("before_action", before_action),
                              ("after_observation", after_observation)):
        errors = validate_agent_submission(submission, phase)
        if errors:
            raise ValueError(f"Invalid {phase} submission: {'; '.join(errors)}")
    trajectory_validator = validators()["trajectory"]
    feedback_validator = trajectory_validator.evolve(
        schema={"$ref": "urn:bioloopbench:schema:common#/$defs/environment_feedback"})
    feedback_errors = [e.message for e in feedback_validator.iter_errors(environment_feedback)]
    if feedback_errors:
        raise ValueError("Invalid environment feedback: " + "; ".join(feedback_errors))
    row = deepcopy({
        "round_index": round_index,
        "hypotheses_before": before_action["hypotheses_before"],
        "action": before_action["action"],
        "observation": environment_feedback["observation"],
        "hypotheses_after": after_observation["hypotheses_after"],
        "revision_summary": after_observation["revision_summary"],
        "evidence_ids": after_observation["evidence_ids"],
        "remaining_budget": environment_feedback["remaining_budget"],
    })
    round_validator = trajectory_validator.evolve(
        schema=trajectory_validator.schema["properties"]["rounds"]["items"])
    round_errors = [e.message for e in round_validator.iter_errors(row)]
    if round_errors:
        raise ValueError("Invalid trajectory round: " + "; ".join(round_errors))
    return row
