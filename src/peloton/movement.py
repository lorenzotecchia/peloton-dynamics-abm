"""Placeholder rider motion: greedy shelter-seeking + forward advance.

Riders are physical objects: they never overlap (rectangular footprint
rider_length x rider_width) and brake to follow the wheel ahead instead of
passing through. The shelter-seeking heuristic remains a stand-in for the
future strategy layer; the no-overlap invariant, however, is permanent physics.
"""

from peloton.physics import exposure_for, overlaps

# Candidate lateral nudges (metres) evaluated each step. 0.0 keeps the line.
_LATERAL_CANDIDATES = (-0.6, -0.3, 0.0, 0.3, 0.6)
_MAX_LATERAL = max(abs(c) for c in _LATERAL_CANDIDATES)

# Max lateral jitter (m). Applied only when it keeps the no-overlap invariant.
_JITTER = 0.09


class _Probe:
    """A throwaway agent used to score exposure at a candidate lateral position."""

    __slots__ = ("pos",)

    def __init__(self, pos):
        self.pos = pos


def next_position(agent, model):
    """Return the agent's next ``(x, y)`` position (never overlapping anyone).

    For each lateral candidate: discard it if the lane change itself would
    overlap someone (staying in line is always feasible); brake so we never
    advance past the wheel of the nearest rider ahead in that lane; then pick
    the lowest-exposure candidate, tie-broken by most forward progress.
    """
    cfg = model.config
    x, y = agent.pos

    advance = cfg.base_speed + model.random.uniform(-cfg.speed_noise, cfg.speed_noise)
    # Anyone who could collide with our final position is within this radius:
    # forward reach + footprint length, plus max lateral shift + jitter + width.
    # max(advance, 0) keeps the radius safe if speed_noise exceeds base_speed.
    search_radius = (
        max(advance, 0.0)
        + cfg.rider_length
        + _MAX_LATERAL
        + _JITTER
        + cfg.rider_width
    )
    others = [
        o
        for o in model.space.get_neighbors(
            agent.pos, radius=search_radius, include_center=True
        )
        if o is not agent
    ]

    def _hits_anyone(pos):
        return any(
            overlaps(
                pos, o.pos,
                rider_length=cfg.rider_length, rider_width=cfg.rider_width,
            )
            for o in others
        )

    best_key = None
    best_x = x
    best_y = y
    for dy in _LATERAL_CANDIDATES:
        cand_y = min(max(y + dy, 0.0), cfg.road_width)

        # Feasibility: the lane change itself must not overlap anyone.
        # dy == 0 always survives because the current position is overlap-free.
        if _hits_anyone((x, cand_y)):
            continue

        # Braking: never advance past the wheel of the nearest rider ahead in lane.
        allowed_x = min(x + advance, cfg.road_length)
        for o in others:
            ox, oy = o.pos
            if ox > x and abs(oy - cand_y) < cfg.rider_width:
                allowed_x = min(allowed_x, ox - cfg.rider_length)
        allowed_x = max(allowed_x, x)        # never forced backward

        probe = _Probe((x, cand_y))
        exposure = exposure_for(
            probe, model,
            draft_radius=cfg.draft_radius, draft_lateral=cfg.draft_lateral,
        )
        # Lowest exposure, then most progress, then hold your line: without the
        # lateral tie-break, full ties always pick the first candidate (-0.6)
        # and the whole field drifts into the road edge.
        key = (exposure, -(allowed_x - x), abs(cand_y - y))
        if best_key is None or key < best_key:
            best_key = key
            best_x = allowed_x
            best_y = cand_y

    # Cosmetic lateral jitter — applied only if it keeps the invariant.
    jittered = best_y + model.random.uniform(-_JITTER, _JITTER)
    jittered = min(max(jittered, 0.0), cfg.road_width)
    if not _hits_anyone((best_x, jittered)):
        best_y = jittered

    return (best_x, best_y)
