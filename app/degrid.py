"""Removes the fine grid that Qwen-Image 2.1's VAE draws over every picture.

The VAE turns each latent pixel into a 16 x 16 block of real pixels and leaves a faint
pattern inside every block: the same small error at the same spot of each block, so it repeats
every 16 pixels (with a 2-pixel checker in it), lined up with the picture's edges. It is 2 to 4
brightness levels (of 255) and shows as a net on smooth areas. It is the same with
full-precision weights and with any tile size, so it comes from the VAE itself.

How it is removed: take the fine detail of the picture, and for each spot inside the 16 x 16
block, average that detail over the smooth parts of the neighbouring blocks. Real detail does
not repeat from block to block, so it averages out; the grid does, so what is left is the grid
in that part of the picture. Subtract it, except on strong edges (letters, outlines), which can look like a repeating
pattern themselves. The grid's strength varies over the picture, and this follows it.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

BLOCK = 16   # pixels per latent pixel (the VAE's spatial compression)
DETAIL = 4   # blur radius that separates fine detail (where the grid is) from shapes and colour
REACH = 2    # neighbouring blocks on each side that are averaged (5 x 5 blocks)
SPREAD = 8.0  # levels a value may differ from the median and still count fully
CAP = 6.0   # brightness levels: the grid is never stronger than this, so no correction is either
QUIET = 30.0  # average fine detail (levels) below which a spot is texture or smooth (fur is 12-26)
BUSY = 50.0   # ... and above which it is a strong edge (letters: 47-80); nothing is changed there


def _box_blocks(x: np.ndarray, r: int) -> np.ndarray:
    """Average over (2r+1) x (2r+1) neighbouring blocks, separately for each spot in the block.
    x has shape (blocks_y, 16, blocks_x, 16, channels)."""
    c = np.cumsum(np.pad(x, ((r + 1, r), (0, 0), (0, 0), (0, 0), (0, 0)), mode="edge"), axis=0)
    x = (c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)
    c = np.cumsum(np.pad(x, ((0, 0), (0, 0), (r + 1, r), (0, 0), (0, 0)), mode="edge"), axis=2)
    return (c[:, :, 2 * r + 1:] - c[:, :, :-2 * r - 1]) / (2 * r + 1)


def _median_blocks(x: np.ndarray, r: int) -> np.ndarray:
    """Median over the (2r+1) x (2r+1) neighbouring blocks, separately for each spot in the
    block. x has shape (blocks_y, 16, blocks_x, 16)."""
    by, _, bx, _ = x.shape
    p = np.pad(x, ((r, r), (0, 0), (r, r), (0, 0)), mode="edge")
    stack = np.stack([p[dy:dy + by, :, dx:dx + bx, :] for dy in range(2 * r + 1) for dx in range(2 * r + 1)])
    return np.median(stack, axis=0)


def _box2d(x: np.ndarray, r: int) -> np.ndarray:
    """Average over a (2r+1) x (2r+1) square around each pixel."""
    for axis in (0, 1):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(x, pad, mode="edge"), axis=axis)
        n = x.shape[axis]
        x = (np.take(c, range(2 * r + 1, 2 * r + 1 + n), axis=axis) - np.take(c, range(n), axis=axis)) / (2 * r + 1)
    return x


def degrid(picture: Image.Image, block: int = BLOCK) -> Image.Image:
    rgb = picture.convert("RGB")
    a = np.asarray(rgb, dtype=np.float32)
    h, w, ch = a.shape
    if h % block or w % block:
        return picture
    smooth = np.asarray(rgb.filter(ImageFilter.GaussianBlur(DETAIL)), dtype=np.float32)
    detail = a - smooth
    # How busy each spot is: the grid alone is a few levels, fur 12-26, letter edges 47-80.
    busy = np.abs(detail).mean(axis=2)
    busy = _box2d(_box2d(busy, 2), 2)
    quiet = (busy < QUIET)[..., None].astype(np.float32)
    # The grid is measured without strong edges. A long straight edge (a letter's stroke) runs
    # through several blocks in a row and would look like a repeating pattern too.
    shape = (h // block, block, w // block, block, ch)
    d = (detail * quiet).reshape(shape)
    q = np.broadcast_to(quiet, a.shape).reshape(shape)
    # First the median of the neighbouring blocks: an edge that got past the mask (an ear's
    # outline) is ignored instead of leaving a ghost of itself in the blocks around it.
    med = np.empty_like(d)
    for c in range(ch):
        med[..., c] = _median_blocks(d[..., c], REACH)
    # The median misses part of the grid; one average around it, where each value may differ
    # from the median by at most SPREAD levels, picks up the rest.
    num = _box_blocks(q * (med + np.clip(d - med, -SPREAD, SPREAD)), REACH)
    den = _box_blocks(q, REACH)
    grid = np.where(den > 0.3, num / np.maximum(den, 1e-6), 0.0)
    grid -= grid.mean(axis=(1, 3), keepdims=True)  # only the pattern inside the block, not its level
    grid = np.clip(grid, -CAP, CAP).reshape(h, w, ch)  # the grid is never stronger than this
    # Not corrected on strong edges: the grid is hidden there and a change risks harm.
    weight = np.clip((BUSY - busy) / (BUSY - QUIET), 0.0, 1.0)[..., None]
    out = a - weight * grid
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
