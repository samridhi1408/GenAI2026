"""Automated generation of interactive troubleshooting tasks from STAR Protocols.

Stages (see docs/taskgen.md):
  corpus    -> retrieve open-access articles for a seed topic and parse them
  extract   -> source-grounded troubleshooting cases, audited by a second model
  siblings  -> group cases by presenting symptom and assemble competing causes
  casefile  -> shared opening, diagnostic-evidence matrix, hidden case notes
  calibrate -> gates that reject leaked, unsolvable or ill-defined instances
Runtime:
  gatekeeper -> answers the agent's free-text experiment requests
  judge      -> blind classification of a diagnosis against the candidate causes
  agent      -> reference agent loop used for calibration and evaluation
"""
