"""Pack detection and pack-level dynamics for the 1-D peloton.

Riders are points on the course (x = distance travelled). A pack is a maximal
run of riders each within ``group_radius`` of the next along x (single-linkage
in 1-D). Within a pack, contribution decides who leads (and so who shelters)
and how fast the whole pack rolls.
"""


def detect_groups(riders, group_radius):
    """Partition ``riders`` into packs by x-proximity. Returns a list of lists.

    Sort by x, then start a new pack whenever the gap to the previous rider
    exceeds ``group_radius``. Transitive: A-B-C all join if each consecutive gap
    is small, even if A and C are far apart.
    """
    ordered = sorted(riders, key=lambda r: r.pos[0])
    groups: list[list] = []
    current: list = []
    for r in ordered:
        if current and r.pos[0] - current[-1].pos[0] <= group_radius:
            current.append(r)
        else:
            if current:
                groups.append(current)
            current = [r]
    if current:
        groups.append(current)
    return groups


def group_speed(members, contribs, cfg) -> float:
    """Pack speed = k_s * sum(C_i * s_m,i) / sum(C_i).

    ``contribs`` is a list of contributions aligned with ``members``. A
    contribution-weighted average of members' threshold speeds, scaled by the
    (non-physical, SA-tuned) pack coefficient k_s. Falls back to the slowest
    rider if nobody contributes (sum C == 0).
    """
    total_c = sum(contribs)
    if total_c <= 0.0:
        return cfg.k_s * min(m.s_m for m in members)
    weighted = sum(c * m.s_m for m, c in zip(members, contribs))
    return cfg.k_s * weighted / total_c


def cohesion_boost(rider, all_riders, cfg) -> float:
    """Mechanic cohesion: extra speed to chase riders visible ahead.

    Looks at all riders within ``cohesion_visibility`` metres ahead on the
    course (x-axis only). The centroid of that visible group determines the
    weight w = d_relative / visibility in [0, 1]; the boost is w * s_m so a
    rider far from the visible centroid (w→1) gets the largest pull, while
    a rider already embedded in the cluster (w≈0) gets almost none.
    """
    vis = cfg.cohesion_visibility
    x_r = rider.pos[0]
    ahead = [r for r in all_riders if r is not rider and 0.0 < r.pos[0] - x_r <= vis]
    if not ahead:
        return 0.0
    x_avg = sum(r.pos[0] for r in ahead) / len(ahead)
    w = (x_avg - x_r) / vis
    if w > 1.0:
        return 0.0
    return w * rider.s_m


def draft_factors(members, contribs, cfg) -> list[float]:
    """Per-rider effective air-drag factor cf_eff in [draft_coefficient, 1].

    The physically-front rider (highest x) always faces full wind (cf=1.0) —
    they cannot avoid it regardless of willingness to cooperate. The remaining
    riders share wind exposure via contribution fractions, so a willing drafter
    who is NOT at the front still takes some wind on behalf of the pack.
    """
    front = max(range(len(members)), key=lambda i: members[i].pos[0])
    total_c = sum(contribs)
    out = []
    for i, c in enumerate(contribs):
        lead = c / total_c if total_c > 0.0 else 1.0 / len(members)
        cf = lead + (1.0 - lead) * cfg.draft_coefficient
        if i == front:
            cf = 1.0   # physical front always in full wind, regardless of contribution
        out.append(cf)
    return out
