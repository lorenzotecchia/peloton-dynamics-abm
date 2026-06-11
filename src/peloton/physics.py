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
