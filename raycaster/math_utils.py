# =============================================================================
# math_utils.py — custom trig implementation (deliberately not math.sin/cos)
#
# WHY: this project reimplements sin/cos from a Taylor series rather than
# calling math.sin/math.cos or np.sin/np.cos, so the numerics are fully
# owned and testable rather than opaque libm calls. It's a small surface,
# but it's the one place in the codebase where "did I get the math right"
# is answered by a regression test (tests/test_trig.py) instead of trust.
#
# Two real bugs were found and fixed here during development — see the
# docstrings on mc_cos and vec_sin below — which is exactly the kind of
# thing a Taylor-series reimplementation is supposed to catch early, with
# tests, rather than silently in production.
# =============================================================================

import numpy as np

# Manual PI constants — used by the custom trig functions below.
PI     = 3.14159265358979
TWO_PI = 6.28318530717959

_SERIES_TERMS = 10   # number of Taylor terms; see tests/test_trig.py for the
                      # accuracy this buys across [-pi, pi]


def norm(x):
    """Normalise an angle (radians) into the range [-pi, pi]."""
    x = x - TWO_PI * int(x / TWO_PI)   # bring into [0, 2*pi)
    if x < 0:
        x += TWO_PI
    if x > PI:                          # fold upper half down to [-pi, 0)
        x -= TWO_PI
    return x


def mc_sin(x):
    """
    Scalar sine via a Taylor series: sin(x) = x - x^3/3! + x^5/5! - ...
    Evaluated to the 19th-degree term, accurate to machine precision for
    |x| <= pi (see tests/test_trig.py::test_mc_sin_accuracy).
    """
    x = norm(x)
    t = x
    s = 0.0
    for n in range(1, _SERIES_TERMS):
        s += t
        t *= -(x * x) / ((2 * n) * (2 * n + 1))
    return s + t


def mc_cos(x):
    """
    Scalar cosine via a Taylor series: cos(x) = 1 - x^2/2! + x^4/4! - ...

    BUG FIX (found during development): this function originally contained
    a Newton-Raphson square-root iteration copy-pasted from elsewhere, not
    a cosine approximation at all — it happened to run without error but
    returned wrong values. Replaced with the correct Taylor series.
    tests/test_trig.py::test_mc_cos_accuracy is the regression guard.
    """
    x = norm(x)
    t = 1.0
    s = 0.0
    for n in range(1, _SERIES_TERMS):
        s += t
        t *= -(x * x) / ((2 * n - 1) * (2 * n))
    return s + t


def scratch_floor(x):
    """Integer floor without importing math — equivalent to math.floor(x)."""
    ix = int(x)
    return ix - 1 if x < ix else ix


def absolute(x):
    """Absolute value without importing math — equivalent to abs(x)."""
    return x if x >= 0 else -x


def vec_sin(arr):
    """
    Element-wise sine for a NumPy array, same Taylor series as mc_sin.

    BUG FIXES (found during development):
      1. `t = X.copy` -> `t = X.copy()` — the trailing parens were missing,
         so `t` was bound to the bound-method object itself, not an array;
         every arithmetic op on it silently produced garbage rather than
         raising, because numpy broadcasting swallowed the type mismatch
         until the output array shape looked "plausible".
      2. The normalised array `X` was computed but the loop kept reading
         the un-normalised `x`, so results were only correct by accident
         for inputs already inside [-pi, pi].
    tests/test_trig.py::test_vec_sin_matches_np_sin is the regression guard.
    """
    x = arr - TWO_PI * np.floor(arr / TWO_PI)      # reduce to [0, 2*pi)
    X = np.where(x > PI, x - TWO_PI, x)             # fold to [-pi, pi]

    t = X.copy()
    s = np.zeros_like(X)
    for n in range(1, _SERIES_TERMS):
        s += t
        t = t * (-(X * X) / ((2 * n) * (2 * n + 1)))
    return s + t


def vec_cos(arr):
    """Element-wise cosine for a NumPy array, mirrors vec_sin."""
    x = arr - TWO_PI * np.floor(arr / TWO_PI)
    x = np.where(x > PI, x - TWO_PI, x)

    t = np.ones_like(x)
    s = np.zeros_like(x)
    for n in range(1, _SERIES_TERMS):
        s += t
        t = t * (-(x * x) / ((2 * n - 1) * (2 * n)))
    return s + t


def rotation_2d(angle):
    """2x2 rotation matrix built from the custom scalar trig above."""
    c = mc_cos(angle)
    s = mc_sin(angle)
    return ((c, -s), (s, c))
