from web import server


def test_ultra_counts_as_extra_pass_access():
    access = server._extra_pass_access("ultra")

    assert access["has_extra_pass"] is True
    assert access["has_ultra"] is True


def test_active_pass_has_premium_without_ultra_access():
    access = server._extra_pass_access("active")

    assert access["has_extra_pass"] is True
    assert access["has_ultra"] is False
