"""Additive two-branch residual preprocessor (Zhao et al. arXiv:2512.15331).

The module tree is PINNED by ``best.pt`` (the additive run's checkpoint,
9,795 parameters, ``load_state_dict(strict=True)`` verified in
``u6_big4/arch.py``): ``spatial_stem 3->16``, ``spatial_residual.body
16->16->16``, ``temporal_stem 24->16`` (8 context frames x 3 channels),
``fusion.gate 32->16->16``, ``to_rgb 16->3``.

The wiring between those modules is NOT pinned — the checkpoint alone cannot
discriminate the activations / fusion algebra / residual form (the MSE gate in
arch.py found 12 variants within 2.7% of the published number), and since this
module is retrained from scratch the wiring is a design choice, not something
to recover. Chosen here (Zhao-faithful reading):

  * ReLU after every conv except the final ``to_rgb``;
  * residual connection in the spatial branch (``sp = stem + body``);
  * sigmoid gate, convex blend fusion ``gate*sp + (1-gate)*tp`` (the paper's
    "conditional attention" between the temporal and spatial branches);
  * temporal context = the CENTER 8 frames of the clip, stacked on channels;
  * output clamped to [0,1].

The model is UNCONDITIONED (no QP input) and UNGATED (no saliency mask) by
design: Zhao train one robust model across the whole rate range, and the
additive edit is allowed to touch every pixel — the accuracy gap is meant to
be >= 0, so there is nothing to protect against. ``cond``/``mask`` are
accepted and ignored so the shared training/eval loops need no branching.

``strength`` scales the residual (``x_pre = x + strength * to_rgb(fused)``).
It is an EVAL-TIME operating point for the gradient-free bit-cost selection
(never a training knob), which is why it is deliberately absent from the
architecture keys restored by ``evaluate()``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditivePreprocessor(nn.Module):
    """Two-branch additive residual editor; 9,795 parameters, fully
    convolutional and resolution-agnostic (trains at 128, evaluates at any
    square size)."""

    def __init__(self, temporal_frames: int = 8, strength: float = 1.0):
        super().__init__()
        if temporal_frames < 1:
            raise ValueError(f"temporal_frames must be >= 1, got {temporal_frames}")
        self.tframes = int(temporal_frames)
        self.strength = float(strength)
        # Key names/shape layout must stay load-compatible with best.pt.
        self.spatial_stem = nn.Conv2d(3, 16, 3, padding=1)
        self.spatial_residual = nn.Module()
        self.spatial_residual.body = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1))
        self.temporal_stem = nn.Conv2d(self.tframes * 3, 16, 3, padding=1)
        self.fusion = nn.Module()
        self.fusion.gate = nn.Sequential(
            nn.Conv2d(32, 16, 1), nn.ReLU(), nn.Conv2d(16, 16, 1))
        self.to_rgb = nn.Conv2d(16, 3, 3, padding=1)
        # Zero-init the output projection so the untrained model IS the identity
        # (out = x + 0). Default PyTorch init starts the residual at RMS
        # 0.10-0.18 (15-20 dB from identity, measured over seeds 0/1/2 with
        # u6_big4/edit_size.py), i.e. the optimiser opens from a random repaint
        # and has to walk back to identity before it can learn anything useful.
        # Standard residual-adapter practice; keys/shapes are unchanged, so
        # best.pt still loads with strict=True.
        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B,3,T,H,W] in [0,1] -> edited clip, same shape, in [0,1].

        ``cond`` (rate condition) and ``mask`` (saliency gate) are accepted for
        loop compatibility and ignored: this model is deliberately
        unconditioned and ungated (see module docstring)."""
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"expected [B,3,T,H,W], got {tuple(x.shape)}")
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)   # [B*T,3,H,W]

        sp = F.relu(self.spatial_stem(frames))
        body = self.spatial_residual.body
        sp = sp + body[2](F.relu(body[0](sp)))

        if t >= self.tframes:
            start = (t - self.tframes) // 2
            ctx = x[:, :, start:start + self.tframes]
        else:  # short clip: pad the temporal context by repeating the last frame
            pad = x[:, :, -1:].expand(b, c, self.tframes - t, h, w)
            ctx = torch.cat([x, pad], dim=2)
        stack = ctx.permute(0, 2, 1, 3, 4).reshape(b, self.tframes * c, h, w)
        tp = F.relu(self.temporal_stem(stack))
        tp = tp.repeat_interleave(t, dim=0)     # clip-level map broadcast over T

        gate = torch.sigmoid(
            self.fusion.gate[2](F.relu(self.fusion.gate[0](torch.cat([sp, tp], dim=1)))))
        fused = gate * sp + (1.0 - gate) * tp
        out = frames + self.strength * self.to_rgb(fused)
        return out.clamp(0.0, 1.0).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
