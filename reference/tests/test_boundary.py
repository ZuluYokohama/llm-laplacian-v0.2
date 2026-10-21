def test_triangle_is_nilpotent():
    pair = triangle_boundaries()
    assert nilpotency_l1(pair["b1"], pair["b2"]) == 0


def test_path_is_nilpotent():
    pair = path_boundaries()
    assert nilpotency_l1(pair["b1"], pair["b2"]) == 0


def test_broken_pair_is_not_nilpotent():
    residual = nilpotency_l1([[1, 0], [0, 1]], [[1, 0], [0, 1]])
    assert residual == 2
