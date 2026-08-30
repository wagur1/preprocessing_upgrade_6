"""Phase-0 CPU smoke for the additive round (RUN_DESIGN_additive.md).

Generates a tiny synthetic Kinetics-style clip set (moving shapes, real mp4
files), builds a hash-split index, runs the REAL training path -- dataset
decode -> _build_models (additive branch + virtual codec + frozen teacher) ->
_fit (Zhao loss, clip_grad, cosine, val) -> checkpoint -- then reloads the
checkpoint through evaluate()'s arch-restore path and checks the loaded model
is the additive one with the intended strength override.

Run:  python scripts/smoke_additive.py [n_steps]
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import apply_overrides, load_config
from src.engine import _build_models, _device, _fit
from src.models.additive import AdditivePreprocessor
from src.tasks.action_recognition import kinetics_category_index
from src.data.prepare_3gb import _canon

ROOT = Path(__file__).resolve().parents[1]


def make_clips(out: Path, n: int = 12) -> list[dict]:
    """Tiny 128x128 mp4s of a bouncing square (structured, cheap to decode)."""
    cls = "abseiling"  # a real Kinetics-400 class so labels map
    label = kinetics_category_index("r3d_18")[_canon(cls)]
    (out / cls).mkdir(parents=True, exist_ok=True)
    recs = []
    for i in range(n):
        path = out / cls / f"clip_{i:03d}.mp4"
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (128, 128))
        x, vx = 16 + (i * 7) % 40, 9
        for f in range(48):
            frame = np.full((128, 128, 3), 24 + (f % 3), np.uint8)
            cv2.rectangle(frame, (x, 48), (x + 32, 80), (200, 90, 40), -1)
            cv2.circle(frame, (64, 100 + (f % 10)), 8, (60, 180, 60), -1)
            w.write(frame)
            x += vx
            if not 8 < x < 88:
                vx = -vx
        w.release()
        recs.append({"path": str(path), "label": label, "class": cls,
                     "bytes": path.stat().st_size})
    return recs


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="u6_smoke_"))
    print(f"[smoke] workspace {tmp}")
    try:
        recs = make_clips(tmp / "data")
        split = {"train": recs[:6], "val": recs[6:9], "test": recs[9:]}
        index = tmp / "index.json"
        index.write_text(json.dumps({
            "meta": {"root": str(tmp / "data"), "n_train": 6, "n_val": 3, "n_test": 3},
            **split}))
        out_dir = tmp / "out"

        cfg = load_config(str(ROOT / "configs" / "additive_ar.yaml"))
        cfg = apply_overrides(cfg, [
            f"out_dir={out_dir}",
            f"data.index={index}",
            "data.frame_size=96",           # CPU speed only
            "train.epochs=3",
            f"train.max_steps={steps}",
            "train.batch_size=2",
            "train.num_workers=0",
            "train.patience=0",
            "train.val_max_batches=2",
            "train.resume=false",
        ])
        cfg["task"]["teachers"] = ["r3d_18"]   # mc3_18 weights not cached locally
        device = _device(cfg)
        print(f"[smoke] device {device}")
        pre, codec, analyzer = _build_models(cfg, device)
        assert isinstance(pre, AdditivePreprocessor), type(pre)

        from torch.utils.data import DataLoader
        from src.data import VideoClipDataset, collate_clips
        ds = VideoClipDataset(str(index), split="train", num_frames=16,
                              frame_size=96, temporal_stride=2, train=True)
        loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0,
                            collate_fn=collate_clips)
        val = DataLoader(VideoClipDataset(str(index), split="val", num_frames=16,
                                           frame_size=96, temporal_stride=2, train=False),
                         batch_size=2, shuffle=False, num_workers=0,
                         collate_fn=collate_clips)

        def prep(batch):
            clips, labels = batch
            return clips.to(device), labels.to(device)

        # first-batch probes: residual live, loss finite, all parts present
        clips, labels = prep(next(iter(loader)))
        x_pre = pre(clips)
        assert x_pre.shape == clips.shape and x_pre.min() >= 0 and x_pre.max() <= 1
        assert not torch.allclose(x_pre, clips, atol=1e-6), "residual must be live"
        assert float((x_pre - clips).abs().mean()) < 0.25, "init edit suspiciously large"

        ckpt = _fit(cfg, pre, codec, analyzer, loader, val, prep,
                    "smoke-additive", n_train=len(ds))
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert torch.isfinite(torch.cat([v.flatten() for v in state["model"].values()
                                         if v.is_floating_point()])).all(), "NaN in weights"

        # evaluate()'s arch-restore path: ckpt cfg wins for arch keys, the
        # strength override (the Phase-2 operating point) must survive.
        from src.engine import evaluate
        eval_cfg = load_config(str(ROOT / "configs" / "additive_ar.yaml"))
        eval_cfg = apply_overrides(eval_cfg, [
            f"data.index={index}", f"out_dir={out_dir}", "data.frame_size=96",
            "eval.held_out_backbone=r2plus1d_18", "model.strength=0.5",
            "eval.include_proxy=true",  # no ffmpeg locally: proxy-only eval
            "eval.qp_list=[50]",
        ])
        evaluate(eval_cfg, ckpt, out_dir=str(out_dir / "eval"))
        ev_pre, _, _ = _build_models(eval_cfg, _device(eval_cfg), role="eval")
        assert isinstance(ev_pre, AdditivePreprocessor)
        assert abs(ev_pre.strength - 0.5) < 1e-9, \
            "eval-time strength override must survive arch-restore"
        print(f"[smoke] eval artifacts: "
              f"{sorted(p.name for p in (out_dir / 'eval').glob('*.json'))}")
        print("smoke passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
