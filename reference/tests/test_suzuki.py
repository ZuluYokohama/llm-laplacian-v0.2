def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def test_recalibration_cuts_info_loss():
    sigma = duplicate_gram()
    raw = info_loss(sigma, 0.5, _identity(6))
    residual = residual_covariance(sigma, [0, 1, 2], 1e-8)
    compensated = info_loss(residual, 0.5, _identity(6))
    assert compensated < raw
    assert compensated < 1e-4
