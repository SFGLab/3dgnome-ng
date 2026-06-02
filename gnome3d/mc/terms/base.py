"""Composeable MC energy terms — the shared abstraction.

An energy *term* (excluded volume, confinement, chain bonds, …) is the unit of
composition.  Each term lives in its own module under `gnome3d.mc.terms` and
provides FOUR implementations plus ONE shared parameter bundle:

  * ``Params``   — a ``namedtuple`` (single source of truth for the term's
                   parameters).  numba consumes it as a typed tuple; JAX consumes
                   it as a pytree (namedtuples are pytrees), so it vmaps for free.
  * ``nb_local`` — njit ``(pos, p, prm) -> float``: the term's energy
                   contribution that involves bead ``p`` (the per-step delta unit).
  * ``nb_init``  — njit ``(pos, prm) -> float``: the term's full score.
  * ``jax_local``— ``(pos, p, p_pos, prm, n_active) -> float`` (jnp).
  * ``jax_init`` — ``(pos, prm, n_active) -> float`` (jnp, padding-insensitive scan).

The numba and JAX bodies are necessarily written separately (njit cannot share a
traced body with XLA), but they sit side by side in one module and are pinned
together by per-term parity tests (``nb == jax`` on random inputs), so drift is a
failing test, not a silent statistical bug.

A *recipe* is an ordered ``tuple[Term, ...]`` — that's what a kernel is (ARCS,
SMOOTH, …).  The JAX driver (`gnome3d.mc.jax_driver.build_mc_kernel`) composes a
recipe by unrolling the term tuple at trace time (a static closure → one fused
XLA program).  numba does NOT recipe-compose: ``@njit`` can't dispatch over a
runtime list of term functions, and the only byte-exact way to generate a
per-recipe kernel is source codegen (rejected as unmaintainable) — a
``literal_unroll`` total reassociates the per-component float sum and breaks
byte-exactness.  So numba keeps its single unified ``_batch_mc_nb`` (one kernel
switched by ``struct_type`` + flags), calling these shared term helpers.  Both
backends thus share the term math + recipe *definition*, not the loop body.

Uniform signatures (some args ignored by a given term, e.g. confinement ignores
``pos``/``p``/``n_active`` in its local form) let the JAX driver treat all
stateless terms identically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from numba import njit as _njit  # type: ignore[reportMissingTypeStubs]

# Typed njit wrapper (mirrors gnome3d.mc.numba.njit) so the term helpers compile
# with identical decorator options and pyright keeps the original signatures.
F = TypeVar("F", bound=Callable[..., Any])


def njit(**kwargs: Any) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        return cast(F, _njit(**kwargs)(fn))

    return decorator


# Standard decorator options shared by every term helper — same as numba.py's,
# so an extracted helper is byte-identical to the inline original it replaced.
NJIT = {"cache": True, "fastmath": True, "nogil": True}


@dataclass(frozen=True)
class Term:
    """One composeable energy term: its name, its four backend implementations,
    its parameter ``namedtuple`` type, and its global delta convention.

    ``delta_factor`` is how the term's local change scales into the global score
    in the MC accept step (2.0 = double-counted pairwise term, 1.0 = per-bead /
    single-counted).  It is applied by the driver to ``(local_new - local_prev)``,
    NOT to the local score itself.
    """

    name: str
    params: type
    nb_local: Callable[..., Any]
    nb_init: Callable[..., Any]
    jax_local: Callable[..., Any]
    jax_init: Callable[..., Any]
    delta_factor: float = 1.0
    # Stateful terms (orientation) carry a mutable cache through the MC loop and
    # can't express their delta as ``local(new) - local(old)``.  When set,
    # ``jax_step(pos, p, old_p, new_p, prm, state, n_active) -> (delta, new_state)``
    # replaces the ``jax_local`` pair in the driver and ``new_state`` is committed
    # on accept.  ``None`` (the default) marks a stateless term.
    jax_step: Callable[..., Any] | None = None
