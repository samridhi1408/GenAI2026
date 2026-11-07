"""Protocol declarations for the environment and simulator interfaces."""

from typing import Any, Mapping, Protocol

Record = Mapping[str, Any]


class Environment(Protocol):
    """Public boundary: supply observations and bookkeeping from trusted state.

    Agent hypotheses and revisions are separate submissions. The environment
    inserts simulator results into trajectory records using make_round.
    """

    def start(self) -> Record:
        """Return public task information and remaining budget, never hidden state."""
        ...

    def step(self, action: Record) -> Record:
        """Accept only an action request; return environment_feedback.

        The environment selects the active hidden instance, obtains its mapped
        result, assigns observation_id and source=environment, and updates budget.
        Experimental observations are never accepted from the agent.
        """
        ...


class Simulator(Protocol):
    """Curator-side boundary, separate from the public environment."""

    def transition(self, hidden_state: Record, action: Record) -> tuple[Record, Record]:
        """Generate (next hidden state, public observation) from trusted inputs.

        The environment supplies hidden state; the agent selects only the action.
        """
        ...
