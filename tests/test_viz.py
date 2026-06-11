from peloton.viz import exposure_to_color


def test_full_shelter_is_green_ish():
    r, g, b = exposure_to_color(0.0)
    assert g > r                       # sheltered -> green dominates

def test_full_exposure_is_red_ish():
    r, g, b = exposure_to_color(1.0)
    assert r > g                       # exposed -> red dominates

def test_color_channels_in_unit_range():
    for e in (0.0, 0.25, 0.5, 0.75, 1.0):
        r, g, b = exposure_to_color(e)
        for c in (r, g, b):
            assert 0.0 <= c <= 1.0
