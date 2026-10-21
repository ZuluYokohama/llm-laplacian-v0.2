def test_phase_boundaries():
    early = gammas(0)
    assert early["phase"] == 1
    assert early["manifold"] == 0.0
    assert early["dirichlet"] == 1e-4
    assert gammas(29999)["phase"] == 1
    assert gammas(29999)["fractional"] == 0.0

    locked = gammas(30000)
    assert locked["phase"] == 2
    assert locked["manifold"] == 0.05
    assert locked["fractional"] == 0.02
    assert locked["hyper"] == 0.01
    assert locked["dirichlet"] == 1e-3

    late = gammas(240000)
    assert late["phase"] == 3
    assert late["manifold"] == 0.0
    assert late["fractional"] == 0.0
    assert late["hyper"] == 0.0
    assert late["dirichlet"] == 1e-4
    assert gammas(300000)["phase"] == 3
