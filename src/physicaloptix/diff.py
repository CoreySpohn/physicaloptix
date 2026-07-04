"""The executable differentiation contract.

``diff_spec`` builds an ``eqx.partition`` / ``eqx.filter_grad`` filter tree
marking exactly the leaves the library treats as differentiable: ``Field.data``
(the chain is linear in the field) and ``ModeBasis.coeffs`` (DM commands, WFE
and drift coefficients). Everything else -- kernels, masks, mode stacks,
spectra -- is constant data: traced, but never differentiated.

Usage::

    params, static = eqx.partition(model, diff_spec(model))
    grads = jax.grad(loss)(params)
"""

import equinox as eqx
import jax

from physicaloptix.core import Field
from physicaloptix.elements.basis import ModeBasis

_HOT = (Field, ModeBasis)


def _node_spec(node):
    """A per-node filter tree: True on the node's hot leaf, False elsewhere."""
    false = jax.tree.map(lambda _: False, node)
    if isinstance(node, Field):
        return eqx.tree_at(lambda f: f.data, false, True)
    return eqx.tree_at(lambda b: b.coeffs, false, True)


def diff_spec(tree):
    """Filter tree marking the differentiable leaves of ``tree``.

    Args:
        tree: Any pytree possibly containing ``Field`` or ``ModeBasis``
            nodes.

    Returns:
        A pytree of booleans with the same structure: ``True`` at
        ``Field.data`` and ``ModeBasis.coeffs`` leaves, ``False`` everywhere
        else.
    """

    def per_leaf(x):
        if isinstance(x, _HOT):
            return _node_spec(x)
        return False

    return jax.tree.map(per_leaf, tree, is_leaf=lambda x: isinstance(x, _HOT))
