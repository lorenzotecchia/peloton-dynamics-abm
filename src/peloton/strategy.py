"""STUB: strategy / game-theory layer.

Real version will return one of {"ride", "cooperate", "defect", "breakaway"}
based on learned probabilities sigma(alpha + beta*d_finish + gamma*E_left).
For the MVP every rider simply rides.
"""


def decide_action(agent, model) -> str:
    """Return the action a rider takes this step. STUB: always ``"ride"``."""
    return "ride"
