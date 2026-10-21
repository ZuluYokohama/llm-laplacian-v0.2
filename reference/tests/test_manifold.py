def _two_clusters():
    return [[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0]]


def test_matched_penalty_is_finite_and_nonnegative():
    points = _two_clusters()
    lap = affinity_laplacian(points, 0.5)
    penalty = manifold_penalty(points, lap)
    assert penalty >= 0.0
    assert penalty < 10.0


def test_crossing_clusters_raises_the_penalty():
    points = _two_clusters()
    lap = affinity_laplacian(points, 0.5)
    crossed = [points[0], points[2], points[1], points[3]]
    assert manifold_penalty(crossed, lap) > manifold_penalty(points, lap)
