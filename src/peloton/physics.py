"""Pure drafting / wind-exposure math. No Mesa dependencies."""


def cf_draft(d_w: float) -> float:
    """Drag multiplier for a rider whose nearest rider ahead is ``d_w`` metres away.

    From the project README: ``CF_draft = 0.62 - 0.0104*d_w + 0.0452*d_w**2``.
    ~0.62 just behind a wheel (max shelter) rising toward 1.0 as the gap grows.
    """
    return 0.62 - 0.0104 * d_w + 0.0452 * d_w**2
