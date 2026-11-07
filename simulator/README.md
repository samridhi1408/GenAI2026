# Simulator interface

A simulator maps hidden state and a permitted diagnostic action to an observation, with a state transition where appropriate.

The Python protocol is declared in `bioloopbench/interfaces.py`. Store curated lookup tables or state-machine definitions here or in private evaluator storage, and reference them from hidden-instance records.
