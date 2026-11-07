# Interaction instructions

You are a molecular biology scientist troubleshooting a molecular cloning experiment.

The cloning workflow has produced an abnormal bacterial transformation outcome. Your goal is to determine the most likely underlying cause using the experimental context and diagnostic evidence available to you.

Read the experimental goal, protocol context, initial observation, permitted diagnostic actions, action budget, and required output fields in `task.json`.

At each round:

1. Generate exactly three distinct hypotheses from the visible experimental context and observations. The benchmark does not supply candidate hypotheses.
2. Rank them in order: rank 1 is most likely, rank 2 is second most likely, and rank 3 is third most likely. Record each hypothesis's `hypothesis_id`, `description`, and `rank`.
3. Submit `hypotheses_before` and one permitted `action` by its action ID, then wait for the environment's response.
4. The environment/simulator generates the experimental observation for your selected action and the active hidden instance. After receiving it, submit `hypotheses_after` with the same or revised set of exactly three hypotheses at ranks 1, 2, and 3, in that order.
5. Briefly explain how the new evidence affected the ranking, including any hypotheses you revised or replaced.

Submit only the agent-authored fields specified in `required_output`: `hypotheses_before` and `action` before feedback, followed by `hypotheses_after`, `revision_summary`, and `evidence_ids` after feedback.

The environment records `observation`, its ID and source, `remaining_budget`, and `round_index` in the trajectory. Do not generate, fill in, or rewrite these fields. The trajectory combines your submissions with environment-generated feedback; you do not author the complete trajectory record.

Record the ranked list before the action in `hypotheses_before` and the updated ranked list after the observation in `hypotheses_after`. Keep a hypothesis's ID when it moves in rank or is clarified without changing its meaning. Use a new ID for a different explanation. Generate all hypothesis descriptions yourself.

Keep the ranking unchanged when the new observation does not justify a change.

Use rank order only. Do not output numeric confidence scores, probabilities, percentages, or categorical confidence levels. A ranking expresses relative plausibility; it does not establish that an explanation is proven.

Diagnostic actions do not need to follow a fixed sequence. Choose the action that you believe will provide the most useful evidence for distinguishing between your current hypotheses.

Use only the permitted diagnostic actions and remain within the assigned action budget.

You may repeat a diagnostic action. Every selection, including a repeat, consumes one action from the budget. Repeating an action returns the same experimental result for the active hidden instance.

Use your biological knowledge to interpret the supplied context and observations. Treat only the initial experimental information and diagnostic observations returned by the environment as observed evidence. Do not assume experimental results that have not been provided.

When you have sufficient evidence, or when the action budget is exhausted, submit:

- `diagnosis`: your final diagnosis, consistent with your latest rank-1 hypothesis.
- `evidence_ids`: the observation IDs supporting your diagnosis. The initial observation has ID `initial_failure`.
- `explanation`: a concise explanation linking the observations to your diagnosis.

Distinguish experimental observations from your interpretation of those observations. Describe unresolved limitations in the explanation without assigning a confidence score or category.
