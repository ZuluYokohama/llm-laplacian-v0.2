def test_leverage_and_sample_stay_in_band():
    edges = [[0, 2, 1.0], [0, 3, 1.0], [1, 2, 1.0], [1, 3, 1.0]]
    probes = [[1.0, -1.0, 0.5, -0.5], [1.0, 1.0, -1.0, -1.0]]
    report = spectral_sparsify(edges, 4, probes, 400, 7)
    assert abs(report["sum_leverage"] - 3.0) < 1e-6
    for edge in report["edges"]:
        assert edge[2] >= 0.0
    for original, sparse in zip(report["original_energy"], report["sparse_energy"]):
        assert original > 0.0
        assert (0.5 * original) <= sparse <= (1.5 * original), (original, sparse)
