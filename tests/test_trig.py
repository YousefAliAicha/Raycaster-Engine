"""
Regression tests for raycaster.math_utils.

Two real bugs were found and fixed in the custom Taylor-series trig during
development:
  1. mc_cos originally contained a Newton-Raphson sqrt iteration copy-pasted
     from elsewhere, not a cosine approximation — it ran without error but
     returned wrong values.
  2. vec_sin had `t = X.copy` (missing parens, binding the method object
     instead of calling it) and ignored the normalised array X in the loop.

These tests exist so those specific failure modes can't silently return.
"""
import math

import numpy as np
import pytest

from raycaster.math_utils import mc_sin, mc_cos, vec_sin, vec_cos, norm, PI


SWEEP = np.linspace(-math.pi, math.pi, 2000)

# Measured with tools/benchmarks.py at 20000 sample points across [-pi, pi]:
#   mc_sin / vec_sin max error: ~5.3e-10
#   mc_cos / vec_cos max error: ~3.5e-9  (worse near +-pi; cosine's Taylor
#   series converges slightly slower there since every term is even-power
#   and the alternating sum cancels less cleanly at the interval edges)
# Thresholds below have headroom over the measured worst case.
SIN_TOL = 2e-9
COS_TOL = 1e-8


def test_mc_sin_accuracy():
    max_err = max(abs(mc_sin(x) - math.sin(x)) for x in SWEEP)
    assert max_err < SIN_TOL, f'mc_sin max error {max_err} across [-pi, pi]'


def test_mc_cos_accuracy():
    max_err = max(abs(mc_cos(x) - math.cos(x)) for x in SWEEP)
    assert max_err < COS_TOL, f'mc_cos max error {max_err} across [-pi, pi]'


def test_mc_cos_is_not_a_sqrt_iteration():
    """Regression guard for bug #1 — a few values that a stray
    Newton-Raphson sqrt loop would get obviously wrong."""
    assert mc_cos(0.0) == pytest.approx(1.0, abs=1e-9)
    assert mc_cos(math.pi) == pytest.approx(-1.0, abs=COS_TOL)
    assert mc_cos(math.pi / 2) == pytest.approx(0.0, abs=1e-9)


def test_vec_sin_matches_np_sin():
    result = vec_sin(SWEEP)
    ref = np.sin(SWEEP)
    max_err = np.max(np.abs(result - ref))
    assert max_err < SIN_TOL, f'vec_sin max error {max_err} across [-pi, pi]'


def test_vec_cos_matches_np_cos():
    result = vec_cos(SWEEP)
    ref = np.cos(SWEEP)
    max_err = np.max(np.abs(result - ref))
    assert max_err < COS_TOL, f'vec_cos max error {max_err} across [-pi, pi]'


def test_vec_sin_returns_array_not_method_object():
    """Regression guard for bug #2 — `t = X.copy` (no call) would raise
    or silently produce a garbage/scalar-ish result under broadcasting."""
    out = vec_sin(np.array([0.0, 1.0, 2.0]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)


def test_vec_sin_normalizes_outside_pi_range():
    """The bug also meant inputs outside [-pi, pi] were evaluated against
    the un-normalised array, only working 'by accident' near zero."""
    x = np.array([10.0, -10.0, 100.0, -100.0])
    result = vec_sin(x)
    ref = np.sin(x)
    assert np.max(np.abs(result - ref)) < 1e-6


def test_norm_wraps_into_pm_pi():
    for raw in [0.0, PI, -PI, 4 * PI, -7.5, 123.456]:
        n = norm(raw)
        assert -PI - 1e-9 <= n <= PI + 1e-9


@pytest.mark.parametrize('x', [0.0, 0.5, 1.0, -1.0, 3.0, -3.0, 10.0, -50.0])
def test_mc_sin_matches_math_sin_outside_principal_range(x):
    assert mc_sin(x) == pytest.approx(math.sin(x), abs=1e-9)


@pytest.mark.parametrize('x', [0.0, 0.5, 1.0, -1.0, 3.0, -3.0, 10.0, -50.0])
def test_mc_cos_matches_math_cos_outside_principal_range(x):
    assert mc_cos(x) == pytest.approx(math.cos(x), abs=COS_TOL)
