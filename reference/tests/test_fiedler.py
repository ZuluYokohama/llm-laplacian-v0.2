def test_path_endpoints_are_peripheral():
    report = fiedler_aei(path_adjacency(6))
    assert report["degenerate"] is False
    assert report["lambda2"] > 1e-3
    assert report["orthogonality"] < 1e-6
    order = sorted(range(6), key=lambda index: report["aei"][index])
    assert set(order[:2]) == {0, 5}, str(report["aei"])


def test_threshold_that_removes_edges_is_degenerate():
    report = fiedler_aei(threshold_adjacency(path_adjacency(6), 1.1))
    assert report["degenerate"] is True
    assert report["lambda2"] < 1e-8
