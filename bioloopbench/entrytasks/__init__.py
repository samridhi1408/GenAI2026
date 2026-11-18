"""Entry tasks: one troubleshooting task per STAR Protocols troubleshooting entry.

The agent is told what went wrong, may ask how the experiment was carried out
(a gatekeeper answers from the protocol text only), and submits a ranked list
of fixes. A fix is correct when it matches one the authors report for that
entry. Nothing the agent is shown or scored against is invented by a model
except paraphrases, which are checked against the source.

Stages (see docs/entrytasks.md):
  corpus    -> shared with bioloopbench.taskgen
  extract   -> one record per qualifying entry, audited by a second model
  calibrate -> judge sanity check and the symptom-only baseline
Runtime:
  env       -> answers questions about the protocol
  judge     -> matches proposed fixes to the reference fixes
  agent     -> reference agent loop (ask, full_protocol, symptom_only)
  evaluate  -> run a model over accepted tasks and aggregate scores
"""
