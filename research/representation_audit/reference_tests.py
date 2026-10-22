def test_compensated_rotation_preserves_linear_predictions_and_gram():
    h, w, r, v = rotation_case()
    assert max(abs(a[0] - b[0]) for a, b in zip(multiply(h, w), multiply(r, v))) < 1e-12
    assert max(abs(x - y) for a, b in zip(gram(h), gram(r)) for x, y in zip(a, b)) < 1e-12


def test_rotation_changes_coordinate_intervention_ranking():
    h, w, r, v = rotation_case()
    original, rotated = coordinate_ablation(h, w), coordinate_ablation(r, v)
    assert original == [6., 12.]
    assert abs(rotated[0] - 15.1008) < 1e-12 and abs(rotated[1] - .9408) < 1e-12
    assert original[0] < original[1] and rotated[0] > rotated[1]


def test_orthogonal_feature_change_preserves_gaussian_energy_in_reference_arithmetic():
    h, _w, r, _v = rotation_case()
    a, b = affinity(h), affinity(r)
    assert max(abs(x - y) for row, other in zip(a, b) for x, y in zip(row, other)) < 1e-12
    assert abs(quadratic(laplacian(a), h) - quadratic(laplacian(b), r)) < 1e-12


def test_fixed_bandwidth_scaling_is_not_affinity_invariant():
    assert affinity([[-1.], [1.]])[0][1] > affinity([[-2.], [2.]])[0][1]
    assert centered_variance([[-2.], [2.]]) == 4 * centered_variance([[-1.], [1.]])


def test_example_permutation_preserves_matched_graph_energy():
    h, _w, _r, _v = rotation_case()
    permuted = [h[i] for i in [2, 0, 3, 1]]
    assert abs(quadratic(laplacian(affinity(h)), h) - quadratic(laplacian(affinity(permuted)), permuted)) < 1e-12


def test_disconnected_self_loop_graph_has_multiple_zero_modes():
    lap = laplacian([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    assert lap == [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    assert quadratic(lap, [[1.], [-2.], [3.]]) == 0


def test_complete_three_node_graph_has_repeated_nonzero_eigenvalues():
    lap = laplacian([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    for vector in [[[1.], [-1.], [0.]], [[1.], [1.], [-2.]]]:
        product = multiply(lap, vector)
        assert max(abs(a[0] - b[0]) for a, b in zip(product, vector)) < 1e-12
    assert abs(quadratic(lap, [[1.], [1.], [1.]])) < 1e-12


def test_incidence_is_coordinate_dependent_but_permutation_equivariant():
    h, _w, r, _v = rotation_case()
    # Support can change even when its aggregate density happens to agree.
    assert [[x > 0 for x in row] for row in h] != [[x > 0 for x in row] for row in r]
    assert positive_density(h) == positive_density([row[::-1] for row in h])
