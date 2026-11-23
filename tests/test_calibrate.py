import numpy as np

from drishyam.calibrate import apply_temperature, fit_temperature


def test_temperature_identity():
    p = np.array([0.8808, 0.1192])
    out = apply_temperature(p, 1.0)
    assert np.allclose(out, p, atol=1e-4)


def _sm(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def test_temperature_recovery():
    """Model raw (T=1) + soft golds generated at T*=2.7 -> fit recovers ~2.7."""
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2, size=(300, 4))
    probs = np.array([_sm(z) for z in logits])
    golds = np.array([_sm(z / 2.7) for z in logits])
    t = fit_temperature(probs, golds, "score")
    assert abs(t - 2.7) < 0.25


def test_temperature_direction():
    """Sharp model + softer golds (overconfident) -> fitted T > 1 (cooling)."""
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2, size=(300, 3))
    sharp = np.array([_sm(z / 0.5) for z in logits])
    soft = np.array([_sm(z / 2.0) for z in logits])
    assert fit_temperature(sharp, soft, "choice") > 1.5
    # inverse: flat model + sharper golds -> T < 1
    flat = np.array([_sm(z / 3.0) for z in logits])
    assert fit_temperature(flat, soft, "choice") < 1.0
