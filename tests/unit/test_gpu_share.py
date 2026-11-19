"""Unit checks for the GPU memory pool of a process that shares the card."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")


from gnome3d.mc.jax.share import GpuShareError, plan_gpu_pool

GIB = 2**30
CARD = 16 * GIB


def raises(env: dict[str, str], gpus: list[tuple[int, int]]) -> str:
    try:
        plan_gpu_pool(env, gpus)
    except GpuShareError as e:
        return str(e)
    return ""


def test_a_free_card_keeps_the_jax_default() -> None:
    p = plan_gpu_pool({}, [(CARD, int(15.4 * GIB))])
    assert p.fraction is None and p.line == "", "a free card keeps the JAX default: " + str(p)


def test_a_card_another_process_holds_gives_a_pool_from_the_free_memory() -> None:
    p = plan_gpu_pool({}, [(CARD, int(3.9 * GIB))])
    want = (3.9 - 0.5) / 16
    assert (
        p.fraction is not None
        and abs(p.fraction - want) < 0.002
        and "XLA_PYTHON_CLIENT_MEM_FRACTION" in p.line
    ), "a card another process holds gives a pool from the free memory: " + str(p)


def test_a_fraction_that_fits_is_kept_and_reported() -> None:
    p = plan_gpu_pool({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.45"}, [(CARD, int(15.4 * GIB))])
    assert p.fraction is None and "7.2 GiB" in p.line, (
        "a fraction that fits is kept and reported: " + str(p)
    )


def test_a_fraction_over_the_free_memory_is_refused() -> None:
    msg = raises({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.45"}, [(CARD, int(3.9 * GIB))])
    assert "0.45" in msg and "3.9 GiB" in msg, "a fraction over the free memory is refused: " + str(
        msg
    )


def test_a_nearly_full_card_is_refused() -> None:
    msg = raises({}, [(CARD, int(0.6 * GIB))])
    assert "0.6 GiB" in msg, "a nearly full card is refused: " + str(msg)


def test_no_preallocation_is_left_alone() -> None:
    p = plan_gpu_pool({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, [(CARD, int(0.6 * GIB))])
    assert p.fraction is None, "no preallocation is left alone: " + str(p)


def test_the_platform_allocator_is_left_alone() -> None:
    p = plan_gpu_pool({"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"}, [(CARD, int(0.6 * GIB))])
    assert p.fraction is None, "the platform allocator is left alone: " + str(p)


def test_no_visible_gpu_changes_nothing() -> None:
    p = plan_gpu_pool({}, [])
    assert p.fraction is None and p.line == "", "no visible GPU changes nothing: " + str(p)


def test_several_gpus_take_the_pool_from_the_fullest_one() -> None:
    p = plan_gpu_pool({}, [(CARD, int(15.4 * GIB)), (CARD, int(5.0 * GIB))])
    assert p.fraction is not None and abs(p.fraction - (5.0 - 0.5) / 16) < 0.002, (
        "several GPUs take the pool from the fullest one: " + str(p)
    )


def test_a_pool_in_gib_becomes_the_fraction_of_the_card() -> None:
    p = plan_gpu_pool({"GNOME3D_GPU_POOL_GIB": "1.5"}, [(CARD, int(15.4 * GIB))])
    assert p.fraction is not None and abs(p.fraction - 1.5 / 16) < 0.001 and "1.5 GiB" in p.line, (
        "a pool in GiB becomes the fraction of the card: " + str(p)
    )
    p = plan_gpu_pool({"GNOME3D_GPU_POOL_GIB": "1.5"}, [(8 * GIB, int(7.6 * GIB))])
    assert p.fraction is not None and abs(p.fraction - 1.5 / 8) < 0.001, (
        "the same pool on an 8 GiB card is twice the fraction: " + str(p)
    )


def test_a_pool_in_gib_over_the_free_memory_is_refused() -> None:
    msg = raises({"GNOME3D_GPU_POOL_GIB": "1.5"}, [(CARD, int(1.2 * GIB))])
    assert "1.5 GiB" in msg and "1.2 GiB" in msg, (
        "a pool in GiB over the free memory is refused: " + str(msg)
    )


def test_a_pool_in_gib_and_a_fraction_together_are_refused() -> None:
    msg = raises(
        {"GNOME3D_GPU_POOL_GIB": "1.5", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.2"},
        [(CARD, int(15.4 * GIB))],
    )
    assert "GNOME3D_GPU_POOL_GIB" in msg and "XLA_PYTHON_CLIENT_MEM_FRACTION" in msg, (
        "a pool in GiB and a fraction together are refused: " + str(msg)
    )
