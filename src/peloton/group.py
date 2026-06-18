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


def draft_factors(members, contribs, cfg) -> list[float]:
    """Per-rider effective air-drag factor cf_eff in [draft_coefficient, 1].

    Returns a list aligned with ``members``. Leadership fraction = C_i / sum(C):
    a rider spends that fraction of the step on the front in full wind (cf=1) and
    the rest sheltered, so cf_eff = lead*1 + (1-lead)*draft_coefficient.
    """
    total_c = sum(contribs)
    out = []
    for c in contribs:
        lead = c / total_c if total_c > 0.0 else 1.0 / len(members)
        out.append(lead + (1.0 - lead) * cfg.draft_coefficient)
    return out
