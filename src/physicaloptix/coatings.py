"""Multilayer coating response: characteristic-matrix method, normal incidence.

Produces the complex dispersion curves that feed ``DispersiveScreen``: a
coating's ``d(log r)/d(thickness)`` curve times a thickness-error map is the
separable chromatic-optic model. Lossless real indices, normal incidence,
scalar (no polarization) -- the intended data path is measured or vendor
phase/group-delay curves; the built-in stack model exists for tests,
sizing, and cases where the layer recipe is actually known.

Sign convention: the library's ``e^{+ikz}`` -- positive optical path is
POSITIVE phase (a matched slab gives ``t = exp(+i 2 pi n d / lambda)``).
The textbook characteristic matrix (Macleod) is the complex conjugate of
the one used here.
"""

import jax
import jax.numpy as jnp


def sellmeier(wavelengths_nm, b, c_um2):
    """Refractive index from the three-term Sellmeier relation.

    Args:
        wavelengths_nm: Wavelengths in nanometers, shape ``(w,)``.
        b: The three B coefficients, shape ``(3,)``.
        c_um2: The three C coefficients in square microns, shape ``(3,)``.

    Returns:
        ``n(lambda)``, shape ``(w,)``.
    """
    lam2 = (jnp.asarray(wavelengths_nm) / 1000.0) ** 2
    terms = b * lam2[:, None] / (lam2[:, None] - c_um2)
    return jnp.sqrt(1.0 + jnp.sum(terms, axis=1))


def multilayer_response(
    wavelengths_nm,
    layer_indices,
    layer_thicknesses_nm,
    *,
    n_incident=1.0,
    n_substrate,
):
    """Complex ``(r, t)`` of a layer stack by the characteristic-matrix method.

    Args:
        wavelengths_nm: Wavelengths in nanometers, shape ``(w,)``.
        layer_indices: Per-layer refractive index, scalar or ``(w,)`` each,
            ordered from the incident side.
        layer_thicknesses_nm: Per-layer physical thickness in nanometers.
        n_incident: Index of the incident medium.
        n_substrate: Index of the substrate, scalar or ``(w,)``.

    Returns:
        ``(r, t)`` complex amplitude coefficients, each shape ``(w,)``, in
        the library's ``e^{+ikz}`` convention.
    """
    wavelengths = jnp.asarray(wavelengths_nm, dtype=float)
    w = wavelengths.shape[0]
    m00 = jnp.ones(w, dtype=complex)
    m01 = jnp.zeros(w, dtype=complex)
    m10 = jnp.zeros(w, dtype=complex)
    m11 = jnp.ones(w, dtype=complex)
    for n_layer, d in zip(layer_indices, layer_thicknesses_nm, strict=True):
        n_l = jnp.broadcast_to(jnp.asarray(n_layer, dtype=complex), (w,))
        delta = 2.0 * jnp.pi * n_l * d / wavelengths
        cos_d, sin_d = jnp.cos(delta), jnp.sin(delta)
        # Conjugate of the textbook matrix: library e^{+ikz} convention.
        a00, a01 = cos_d, -1j * sin_d / n_l
        a10, a11 = -1j * n_l * sin_d, cos_d
        m00, m01, m10, m11 = (
            m00 * a00 + m01 * a10,
            m00 * a01 + m01 * a11,
            m10 * a00 + m11 * a10,
            m10 * a01 + m11 * a11,
        )
    n0 = jnp.broadcast_to(jnp.asarray(n_incident, dtype=complex), (w,))
    ns = jnp.broadcast_to(jnp.asarray(n_substrate, dtype=complex), (w,))
    b = m00 + m01 * ns
    c = m10 + m11 * ns
    r = (n0 * b - c) / (n0 * b + c)
    t = 2.0 * n0 / (n0 * b + c)
    return r, t


def thickness_kernel(
    wavelengths_nm,
    layer_indices,
    layer_thicknesses_nm,
    layer,
    *,
    output="r",
    n_incident=1.0,
    n_substrate,
):
    """``d(log r or log t) / d(thickness_nm)`` of one layer, shape ``(w,)``.

    The separable chromatic-optic ingredient: this curve is a
    ``DispersiveScreen`` kernel row, and the layer's thickness-error map is
    the matching mode in ``B`` (coefficient in the same nm unit).

    Args:
        wavelengths_nm: Wavelengths in nanometers, shape ``(w,)``.
        layer_indices: Per-layer refractive index, scalar or ``(w,)`` each,
            ordered from the incident side.
        layer_thicknesses_nm: Per-layer physical thickness in nanometers.
        layer: Index into ``layer_indices``/``layer_thicknesses_nm`` of the
            layer whose thickness the derivative is taken with respect to.
        output: Which amplitude to differentiate, ``"r"`` or ``"t"``.
        n_incident: Index of the incident medium.
        n_substrate: Index of the substrate, scalar or ``(w,)``.

    Returns:
        The complex log-derivative curve, shape ``(w,)``.

    Raises:
        ValueError: If ``output`` is not ``"r"`` or ``"t"``.

    Note:
        Singular where the chosen output amplitude approaches zero (an AR
        point or a dichroic's transmission/reflection null). The first-order
        ``exp(D*B)`` extrapolation this kernel feeds is valid only for
        thickness errors small against the curve's variation scale.
    """
    if output not in ("r", "t"):
        raise ValueError(f"output must be 'r' or 't', got {output!r}")
    thicknesses = [jnp.asarray(d, dtype=float) for d in layer_thicknesses_nm]

    def log_response(d_layer):
        stack = list(thicknesses)
        stack[layer] = d_layer
        r, t = multilayer_response(
            wavelengths_nm,
            layer_indices,
            stack,
            n_incident=n_incident,
            n_substrate=n_substrate,
        )
        return jnp.log(r if output == "r" else t)

    # Forward-mode handles real-input/complex-output directly (it is
    # grad/jacrev that would raise): jax.jacfwd of a real-input,
    # complex-output function is a well-defined JVP.
    return jax.jacfwd(log_response)(thicknesses[layer])
