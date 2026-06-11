"""Placeholder rider motion: greedy shelter-seeking + forward advance.

This is a heuristic stand-in. The strategy layer will eventually drive positioning
(lead-outs, team tactics); for the MVP riders just tuck toward the lowest-exposure
lateral spot near them.
"""

from peloton.physics import exposure_for

# Candidate lateral nudges (metres) evaluated each step. 0.0 keeps the line.
_LATERAL_CANDIDATES = (-0.6, -0.3, 0.0, 0.3, 0.6)


class _Probe:
    """A throwaway agent used to score exposure at a candidate lateral position."""

    __slots__ = ("pos",)

    def __init__(self, pos):
        self.pos = pos


def next_position(agent, model):
    """Return the agent's next ``(x, y)`` position."""
    cfg = model.config
    x, y = agent.pos

    # Forward advance with uniform noise.
    advance = cfg.base_speed + model.random.uniform(-cfg.speed_noise, cfg.speed_noise)
    new_x = x + advance

    # Greedy shelter-seeking: pick the candidate lateral offset with lowest exposure,
    # evaluated at the CURRENT x against current neighbour positions.
    best_y = y
    best_exposure = None
    for dy in _LATERAL_CANDIDATES:
        cand_y = min(max(y + dy, 0.0), cfg.road_width)
        probe = _Probe((x, cand_y))
        exp = exposure_for(
            probe,
            model,
            draft_radius=cfg.draft_radius,
            draft_lateral=cfg.draft_lateral,
        )
        if best_exposure is None or exp < best_exposure:
            best_exposure = exp
            best_y = cand_y

    # Small lateral noise, then clamp into the road.
    new_y = best_y + model.random.uniform(-0.1, 0.1)
    new_y = min(max(new_y, 0.0), cfg.road_width)

    return (new_x, new_y)
