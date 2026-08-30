"""Checks for the additive two-branch preprocessor (needs torch)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.additive import AdditivePreprocessor

BEST_PT = Path(r"C:\Users\Wagur1\Downloads\preprocessing-final-results\checkpoints\best.pt")


def main() -> None:
    # (a) exact parameter count: the additive run's checkpoint has 9,795.
    pre = AdditivePreprocessor()
    n = sum(p.numel() for p in pre.parameters())
    assert n == 9795, f"param count {n} != 9795"

    # (b) structural pin: if the additive run's best.pt is on this machine,
    #     our module tree must strict-load it (key names + every shape).
    if BEST_PT.exists():
        sd = torch.load(BEST_PT, map_location="cpu", weights_only=False)["preprocessor"]
        pre.load_state_dict(sd, strict=True)
        print(f"[test_additive] strict-loaded best.pt ({n} params)")
    else:
        print("[test_additive] best.pt not found locally -- skipping strict-load pin")

    # (c) shape / range / resolution / clip-length handling.
    for t, hw in ((16, 128), (16, 224), (4, 96)):
        x = torch.rand(2, 3, t, hw, hw)
        y = pre(x)
        assert y.shape == x.shape, (y.shape, x.shape)
        assert y.min() >= 0.0 and y.max() <= 1.0, "output must be clamped to [0,1]"

    # (d) unconditioned and ungated: cond/mask must not change the output.
    x = torch.rand(2, 3, 16, 64, 64)
    y0 = pre(x)
    y1 = pre(x, torch.zeros(2, 1))
    y2 = pre(x, torch.zeros(2, 1), torch.ones(2, 1, 16, 64, 64))
    assert torch.equal(y0, y1) and torch.equal(y0, y2), "cond/mask must be ignored"

    # (e) gradient flows to every parameter of every branch.
    x = torch.rand(2, 3, 16, 64, 64)
    pre(x).mean().backward()
    for name, p in pre.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"no/NaN grad: {name}"
    pre.zero_grad(set_to_none=True)

    # (f) strength is the operating point: 0 -> exact identity; s -> x + s*(y1 - x)
    #     (checked away from the clamp with a tiny-residual instance).
    tiny = AdditivePreprocessor()
    with torch.no_grad():
        tiny.to_rgb.weight.mul_(0.01)
        tiny.to_rgb.bias.mul_(0.01)
    x = torch.rand(1, 3, 16, 64, 64) * 0.4 + 0.3
    with torch.no_grad():
        y1 = tiny(x)
        tiny.strength = 0.5
        yh = tiny(x)
        tiny.strength = 0.0
        y0 = tiny(x)
    assert torch.equal(y0, x), "strength=0 must be exact identity"
    assert torch.allclose(yh, x + 0.5 * (y1 - x), atol=1e-6), \
        "strength=s must scale the residual linearly"

    # (g) not an identity at random init: the additive residual is live.
    with torch.no_grad():
        d = (AdditivePreprocessor()(x) - x).abs().mean().item()
    assert d > 1e-6, "random-init residual must be non-zero"

    print("additive self-check passed")


if __name__ == "__main__":
    main()
