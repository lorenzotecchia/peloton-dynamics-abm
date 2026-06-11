"""Pure drafting / wind-exposure math. No Mesa dependencies."""


def cf_draft(d_w: float) -> float:
    """Drag multiplier for a rider whose nearest rider ahead is ``d_w`` metres away.

    From the project README: ``CF_draft = 0.62 - 0.0104*d_w + 0.0452*d_w**2``.
    ~0.62 just behind a wheel (max shelter) rising toward 1.0 as the gap grows.
    """
    return 0.62 - 0.0104 * d_w + 0.0452 * d_w**2


def neighbors_ahead(agent, model, *, draft_radius: float, draft_lateral: float):
    """Riders directly in front of ``agent`` within the drafting cone.

    A neighbour counts when it is strictly ahead (greater x), within
    ``draft_radius`` longitudinally, and within ``draft_lateral`` laterally.
    """
    x, y = agent.pos
    # Search radius covers the longitudinal range; we filter the cone precisely below.
    candidates = model.space.get_neighbors(
        agent.pos, radius=draft_radius, include_center=True
    )
    ahead = []
    for other in candidates:
        if other is agent:
            continue
        ox, oy = other.pos
        if ox <= x:
            continue                          # not ahead
        if ox - x > draft_radius:
            continue                          # too far forward
        if abs(oy - y) > draft_lateral:
            continue                          # outside lateral cone
        ahead.append(other)
    return ahead


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def exposure_for(agent, model, *, draft_radius: float, draft_lateral: float) -> float:
    """Wind exposure of ``agent`` in ``[0, 1]``.

    ``0.0`` = fully sheltered, ``1.0`` = fully in the wind. Computed from the
    nearest rider ahead inside the draft cone: a low ``cf_draft`` (deep shelter)
    yields a large saving and therefore low exposure.
    """
    ahead = neighbors_ahead(
        agent, model, draft_radius=draft_radius, draft_lateral=draft_lateral
    )
    if not ahead:
        return 1.0

    x = agent.pos[0]
    nearest_gap = min(other.pos[0] - x for other in ahead)
    # Normalise cf_draft into [0,1]: 0.62 (max shelter) -> 0.0, 1.0 (no shelter) -> 1.0
    _CF_MIN = 0.62
    raw = (cf_draft(nearest_gap) - _CF_MIN) / (1.0 - _CF_MIN)
    return _clamp01(raw)
