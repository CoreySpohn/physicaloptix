"""Cross-check the coating module against an independent implementation."""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.coatings import multilayer_response


def test_bare_interface_matches_hcipy_fresnel():
    """Normal-incidence Fresnel coefficient vs hcipy's, several index pairs."""
    hcipy = pytest.importorskip("hcipy")
    for n1, n2 in [(1.0, 1.5), (1.0, 2.35), (1.38, 2.35)]:
        r_ours, _ = multilayer_response(
            jnp.array([550.0]), [], [], n_incident=n1, n_substrate=n2
        )
        r_s, _ = hcipy.fresnel_reflection_coefficients(n1, n2, 0.0)
        np.testing.assert_allclose(complex(r_ours[0]), complex(r_s), rtol=1e-10)
