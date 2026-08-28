# preprocessing_upgrade_6

**Universal, analyzer-agnostic video preprocessing for Video Coding for Machines (VCM), in front of a *frozen* standard codec (x264 / x265).**

Only a small preprocessor network is trained. The video codec and every downstream
vision model ("analyzer") stay frozen and standard — no bitstream changes, no
decoder changes, no per-CTU QP maps. The preprocessor edits pixels *before*
encoding so that, at the same bitrate, a machine still sees what it needs.

> 🇻🇳 **Tóm tắt.** Bản nâng cấp 6 của pipeline tiền xử lý VCM. Chỉ huấn luyện
> *preprocessor* đặt **trước** codec tiêu chuẩn (x264/x265) **đóng băng**.
> Toàn bộ hạ tầng 5.1 (A1 multi-teacher, A2 task-mask, A3 STE, C1 yuv420 proxy,
> C2 in-grid QP) được giữ nguyên. Đóng góp mới **(D1) structural saliency
> gating**: edit pixel bị nhân với `(1 − M)` theo saliency của analyzer đóng
> băng — `x_pre = x + (1 − M)·edit` — nên **vùng task-critical là identity
> chính xác theo cấu trúc**, không loss nào đổi được accuracy của nó lấy rate.

```
                     (trained)              (FROZEN)                 (FROZEN panel)
 video x ─► Preprocessor θ ─► x_pre ─► Standard codec ─► x̂ ─► Analyzer(s) ─► machine label
            U-Net + FiLM(rate)          x264 / x265                r3d_18 / mc3_18 / …
            + SFT(motion)               (yuv420 proxy at train)    + a HELD-OUT analyzer at eval
            + D1 GATE: edit *= (1-M)    M = task saliency of the frozen analyzer
```

The claim we optimize is the **preprocessor gain on the *same* codec**:
`BD-Rate(prep+x265 vs x265)` and `BD-Rate(prep+x264 vs x264)` — negative means the
preprocessor lets the machine reach the same accuracy at fewer bits. (Cross-codec
comparisons are reported too, but QP is not comparable across codecs, so they are
reference-only.)

---

## Why upgrade-6 (what ended upgrade-5.1)

5.1's first two full 3-seed runs came out **negative** (`prep+h264 +2.1 %`,
`prep+h265 +3.2 %`; replicated +9.6 %/+12.2 % — STE stage flips between
near-identity and kept-destructive per seed). A systematic 11-variant
loss-weight campaign (2026-08-27/28, single seed, Stage-1 only) then mapped the
whole reachable space:

| variant | change | mean QP30-gap | outcome |
|---|---|---|---|
| control | μ=10, λ_task=1, β=0.01 | −0.310 | pathology: edits cut bpp by destroying ~30 acc points at QP30 |
| mu3 / task3 / beta003 | μ→3 / λ_task→3 / β→0.003 | −0.254 / −0.333 / −0.295 | no single knob closes the gap |
| flatkill (γ=δ=0) | remove flattening pressures | −0.321 | not the cause |
| **distill2 (ω=2)** | raise distillation | **−0.213** | best of all; still fails −0.05 |
| distill5 (ω=5) | distillation saturation | −0.305 | ω peaks at 2 |
| noqp50 (drop QP50 from train) | curriculum | −0.317 | dead-QP curriculum doesn't rescue |
| zhao-bridge (μ=10, β=0.001, γ=δ=0 — the paper's exact loss) | transfer test | −0.298, BD +87 %/+60 % | **Zhao's loss does not transfer to our infra** |

Two mechanisms were established: (i) the task cross-entropy **dies to random-guess**
under proxy-codec damage at heavy QPs (train logs: task ≈ ln 400), so during most
of training there is *no accuracy-preserving gradient*; (ii) with no live task
signal, the remaining pressures steer the edit into "cheap for the codec, unreadable
for the analyzer". Every loss-side fix was tried and failed ⇒ the failure is
**structural**.

**D1 removes the failure mode by construction.** The edit is gated per-pixel by the
frozen analyzer's saliency: task-critical pixels are an exact identity — no loss
term can trade their accuracy for rate — and the network is free to spend its
creativity on the background, where bits are wasted anyway. This is the
pixel-domain analogue of prompt-guided prefiltering (Azizian & Bajić, ICME 2026,
arXiv:2604.00314: keep task-relevant regions, smooth the rest, 25–50 % savings
with unchanged accuracy) and inherits the A2 saliency machinery already in the
harness — promoted from a *loss reweighting* (a pressure the optimizer can
override) to a *forward-pass structure* (which it cannot).

Design invariants:

* loss weights in `configs/universal_action_recognition.yaml` are **identical to
  5.1** — the gate is the single changed variable, so any delta vs the 11-variant
  campaign attributes to D1;
* the mask is **detached** (no second-order gradients), cheap (one extra backward
  through the frozen analyzer, already computed for A2);
* at eval the gate mask comes from the **eval analyzer** (held-out backbone) —
  train/eval consistency, and the honest "universal" reading: protection regions
  derived from whichever frozen analyzer will consume the stream;
* `model.gate: false` reproduces 5.1 exactly (ablation arm).

## Why upgrade-3 (what was limiting upgrade-2)

The direct baseline is **Zhao et al., "A Preprocessing Framework for Video Machine
Vision under Compression," arXiv:2512.15331 (2025)** — a neural preprocessor
trained through a differentiable *virtual codec* and deployed in front of real
x264/x265, reporting **> 15 % BD-Rate** savings across action-recognition and
tracking backbones.

Our `upgrade2` reproduced the *harness* (differentiable proxy, real-codec eval,
BD-Rate-on-accuracy) and reached a **working but modest** operating point:
in-domain `prep+compressai` ≈ −2 to −6 % with positive BD-accuracy, but transfer
to real x264/x265 stalled near break-even. Its own report (`docs/bao_cao_preprocessing.md`)
diagnosed three ceilings:

| Limitation in upgrade-2 | Fixed in upgrade-3 by |
|---|---|
| Trained against **one** frozen analyzer → edit overfits that network | **A1** multi-teacher panel + held-out generalization test |
| Bit/edit budget spent **uniformly** → cutting background also blurs the object | **A2** task-importance spatial mask |
| Trained **entirely** through a mismatched proxy → poor transfer to real x26x | **A3** real-codec-in-the-loop calibration + soft→hard quant anneal |

## Why upgrade-5.1 (what was still limiting upgrade-3)

The measured upgrade-3 result (1000-step STE run, seed 0, held-out analyzer):
`prep+h264 −1.61 %`, `prep+h265 −0.27 %` BD-Rate — transfer turned negative
but remained an order of magnitude short of the paper's −12…−19 %. Two
signatures in the rate-accuracy curves motivated the two 5.1 contributions:

| Remaining limitation in upgrade-3 | Fixed in 5.1 by |
|---|---|
| Proxy codes **RGB planes**; real codecs code **YCbCr 4:2:0** — chroma is halved at *every* QP and quantised coarser, so the training rate-gradient under-charges chroma-heavy edits and never rewards moving budget to luma | **C1** yuv420 colourspace proxy (`src/models/color.py`, `VirtualCodec(colorspace="yuv420")`) |
| FiLM conditions extrapolated at heavy QPs (train grid ≠ eval grid; `prep+h265` accuracy *collapsed* at QP50) | **C2** in-grid QP protocol: train exactly on the eval QPs `[30, 35, 40, 45, 50]` |

## The three upgrade-3 contributions

### A1 — Universal, analyzer-agnostic preprocessor (headline)
`src/tasks/multi_teacher.py`, `src/tasks/base.py::build_analyzer`

Instead of a single frozen analyzer, the preprocessor is trained against a **panel
of frozen teachers** (`task.teachers`, e.g. `r3d_18` + `mc3_18`). Each step either
averages the task loss over all teachers (`mean`) or samples one teacher
(`sample`, a stochastic-multi-teacher regularizer that stops the edit from
specializing to a single network). Feature distillation follows the *active*
teacher so it stays coherent within a step.

The generalization claim is then measured against a **held-out analyzer that is
never in the panel** (`eval.held_out_backbone`, e.g. `r2plus1d_18`). Standard-codec
preprocessing works (Lu et al. 2206.05650) only show *narrow* same-family transfer;
learned-codec works that do prove broad held-out transfer (**UG-ICM**,
arXiv:2501.04579; **All-in-One Transfer**, arXiv:2504.12997) retrain the codec.
"Universal preprocessing proven on a broad held-out analyzer *with the standard
codec left frozen*" is the open niche this repo targets. Multi-teacher aggregation
follows multi-teacher distillation (arXiv:2510.18680) and the feature-modulation
multi-task preprocessor of **Yang et al., TCSVT 2024** (DOI 10.1109/TCSVT.2023.3348995).

### A2 — Task-importance spatial mask
`src/models/task_mask.py`, `src/losses.py`

A **gradient-saliency** map of the task loss w.r.t. the input, `m = |∂L_task/∂x|`,
computed with one extra backward through the frozen teacher and then **detached**.
It *spatially reweights* the edit (`delta`) and total-variation (`gamma`) penalties
by `1 − m`, so the preprocessor smooths and stops spending bits on **background**
while sparing the object the machine needs. This is the pixel-domain, differentiable
analogue of task-driven bit allocation — cf. **Reinforced Bit Allocation**
(arXiv:1910.07392), **feature-preserving RDO** (arXiv:2504.02216), and **ROI
retargeting for machines** (EURASIP JIVP 2025, DOI 10.1186/s13640-025-00682-3) —
but without touching the encoder. It turns upgrade-2's *global* in-domain↔transfer
knob (`gamma`) into a *spatial* one.

### A3 — Real-codec calibration (proxy → real transfer)
`src/models/ste_codec.py`, `src/models/virtual_codec.py`

Two mechanisms that close the proxy→real gap that capped upgrade-2's transfer:

1. **Straight-through real codec.** `STECodec` runs the *real* x264/x265 in the
   forward pass and borrows the differentiable proxy's gradient in the backward
   pass: `x̂ = x_proxy + (x_real − x_proxy).detach()`, and likewise for bpp. The
   loss the optimizer sees is computed on the **true** reconstruction and **true**
   coded rate. This is the single most reliable transfer recipe in the literature:
   **Lu et al. (arXiv:2206.05650 / TCSVT 2024)** measured forward-real-codec at
   −20.3 % BD-Rate vs −14.6 % for a proxy used in both directions. (ffmpeg-per-step
   is slow → used as a short **calibration fine-tune** on top of a proxy-pretrained
   checkpoint; see the two-stage recipe below.)
2. **Soft→hard quantizer annealing.** The block-DCT virtual codec anneals its
   training quantizer from additive-uniform noise (soft) to straight-through hard
   rounding (`codec.anneal: 1.0`), so the proxy ends at the codec's real hard
   quantizer — cf. the soft-quantizer α→∞ schedule of **J4D** (arXiv:2606.16185).

The block-DCT proxy geometry itself (vs a learned wavelet proxy) follows Zhao et al.
2512.15331 and **Sandwiched Compression** (arXiv:2402.05887); the virtual-codec +
straight-through idea traces to **DPP** (Chadha & Andreopoulos, CVPR 2021) and the
differentiable-JPEG proxy of **Talebi et al.** ("Better Compression with Deep
Pre-Editing," IEEE TIP 2021).

## The two upgrade-5.1 contributions

### C1 — yuv420 colourspace proxy (the colourspace gap STE cannot close)
`src/models/color.py`, `src/models/virtual_codec.py`

STE corrects the *value* the loss sees, but the gradient still flows through the
proxy — wrong geometry still gives a wrong direction. x264/x265 never code RGB:
they convert to BT.601 YCbCr, subsample chroma 2×2 (**at every QP**, independent
of rate), and quantise chroma coarser (the H.26x chroma QP offset ≈ +6 QP at
QP50). An RGB proxy therefore under-charges chroma-heavy edits: the
preprocessor keeps spending budget on chroma detail the real codec destroys for
free. The 5.1 virtual codec reproduces the whole geometry —

```
RGB ─► BT.601 YCbCr ─► chroma 2×2 down ─► per-plane DCT+quant (chroma_step× coarser)
  ─► chroma upsample ─► YCbCr → RGB
```

— as differentiable ops, so the training rate/distortion gradients finally
match the deployment codec's colourspace damage. `colorspace: rgb` keeps the
legacy path for the ablation table.

### C2 — in-grid QP protocol
`configs/*.yaml` (`train.qp_list = eval.qp_list = [30, 35, 40, 45, 50]`)

The FiLM rate-conditioning is only trained on the QPs it sees; eval QPs outside
the training grid are extrapolations (the upgrade-3 `prep+h265` accuracy
*collapse at QP50* is the visible symptom). 5.1 trains on exactly the eval grid
with five distinct proxy qualities — the paper's own QP range — removing the gap
at zero cost.

---

## Repository layout

```
src/
  models/
    preprocessor.py    U-Net + FiLM(rate) + SFT(motion) pixel editor (trained)
    virtual_codec.py   differentiable block-DCT proxy (+ A3 soft→hard anneal)
    codec.py           CompressAI learned proxy (alternative training codec)
    ste_codec.py       A3 straight-through real-codec wrapper
    task_mask.py       A2 gradient-saliency importance map + masked TV
  tasks/
    base.py            TaskAnalyzer interface + build_task / build_analyzer
    multi_teacher.py   A1 frozen-teacher panel
    action_recognition.py  Kinetics-400 video classifiers (r3d_18/mc3_18/r2plus1d_18)
    tracking.py, siamfc.py, pytracking_adapter.py  GOT-10k tracking task
  codecs/standard.py   real x264/x265 via ffmpeg (honest coded bpp)
  losses.py            L_task + ω·L_distill + β·bpp + τ·L_temp (+ δ,γ masked) (+ μ·L_D)
  metrics/bd_rate.py   Bjøntegaard BD-Rate/BD-accuracy on the rate–accuracy curve
  engine.py            train / eval loop, rate conditioning, 6-pipeline BD-Rate
configs/
  universal_action_recognition.yaml   A1+A2+A3 headline config
  action_recognition.yaml, tracking.yaml   single-analyzer baselines
docs/MODEL.md, docs/IMPROVEMENTS.md, docs/KAGGLE.md
kaggle/   ready-to-run Kaggle notebook + launcher
```

Every non-trivial module has a `__main__` self-check (`python -m src.models.virtual_codec`,
`python -m src.tasks.multi_teacher`, `python -m src.models.task_mask`,
`python -m src.models.ste_codec`, `python -m src.metrics.bd_rate`).

---

## Quickstart

### Kaggle one-shot

Attach `rohanmallick/kinetics-train-5per`, enable GPU + Internet, then run one cell:

```bash
%%bash
set -euo pipefail
cd /kaggle/working
if [ -d pre_processing_upgrade_3/.git ]; then
  git -C pre_processing_upgrade_3 pull --ff-only
else
  git clone -q https://github.com/wagur1/pre_processing_upgrade_3.git
fi
cd pre_processing_upgrade_3
bash kaggle/run.sh
```

`kaggle/run.sh` detects the mounted Kinetics directory and rebuilds legacy indexes
that do not contain the independent `test` split.

### Manual run

```bash
pip install -r requirements.txt          # torch, torchvision, compressai, opencv, ffmpeg on PATH

# 1) build a data index (see docs/KAGGLE.md for Kinetics / GOT-10k prep)
python -m src.data.prepare_3gb   --help
python -m src.data.prepare_got10k --help

# 2a) STAGE 1 — proxy pretrain (fast, differentiable block-DCT proxy)
python train.py --config configs/universal_action_recognition.yaml \
    data.index=data/index/kinetics_3gb.json train.epochs=5

# 2b) STAGE 2 — real-codec calibration fine-tune (A3; short, ffmpeg-in-the-loop)
python train.py --config configs/universal_action_recognition.yaml \
    data.index=data/index/kinetics_3gb.json \
    codec.kind=ste codec.ste_codec=h265 train.finetune=true train.resume=false \
    train.epochs=6 train.lr=3e-5           # a few hundred extra steps

# 3) evaluate on a HELD-OUT analyzer, real x264/x265 anchors, BD-Rate
python evaluate.py --config configs/universal_action_recognition.yaml \
    --ckpt outputs/checkpoints/preprocessor.pth \
    data.index=data/index/kinetics_3gb.json eval.split=test eval.held_out_backbone=r2plus1d_18
```

The preprocessor checkpoint stores **only** the preprocessor weights, so it is
codec-agnostic: evaluate the same checkpoint against any codec by changing
`codec.kind` / the anchor list. Outputs (`results.json`, `curves.csv`,
`rate_accuracy.png`, `qualitative.png`) land in `outputs/eval/`.

With `eval.per_sequence: true` (the default action-recognition setting), the
evaluator also writes `sequence_points.csv`, `sequence_bd_rate.csv`, and
`sequence_bd_rate.json`. These contain the five QP points and same-codec
`prep+h264 vs h264` / `prep+h265 vs h265` BD-Rate for each held-out video.
Per-video `top1` is retained as a binary diagnostic; because a binary value
does not provide a useful five-point curve for most individual videos, the
per-sequence BD fit uses `target_prob`, the frozen analyzer's probability for
the ground-truth class. The aggregate headline remains BD-Rate on dataset
top-1 accuracy.

---

## Evaluation protocol & metric

`src/metrics/bd_rate.py` computes **BD-Rate with machine accuracy as the quality
axis** (top-1 for recognition, success-plot AUC for tracking) instead of PSNR,
integrating `log(rate)` over the overlapping accuracy range (Bjøntegaard). By
default, evaluation traces `prep+{x264,x265}` vs bare `{x264,x265}` and reports:

* **`bd_prep_gain`** — same-codec preprocessor gain (`prep+x265 vs x265`, …). **The real claim.**
* `bd_vs_anchor` — cross-codec (reference only; QP not comparable across codecs).

Set `eval.include_proxy=true` to add virtual/CompressAI diagnostic curves.

Negative BD-Rate = fewer bits at equal accuracy. See [`docs/MODEL.md`](docs/MODEL.md)
for the full architecture and [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md) for the
line-by-line delta vs the baseline and upgrade-2.

---

## Datasets

* **Kinetics-400** (Kay et al., 2017) — action recognition; frozen torchvision
  video ResNets carry the canonical 400-class ordering.
* **GOT-10k** (Huang et al., TPAMI 2021) — single-object tracking; success-plot
  AUC / AO / SR. Default tracker is a self-contained SiamFC; the paper's exact
  KYS/DiMP/ATOM/PrDiMP run via `pytracking` (`scripts/install_pytracking.sh`).

## Reproducibility notes

* This repo is a **research harness**, not a set of frozen numbers. BD-Rate on a
  small Kinetics subset fluctuates ±3–4 % per seed — run 3–5 seeds and report a CI
  (`kaggle/report_ci.py`). Effect size and significance grow with data, not with
  architecture; A1/A2/A3 raise the *ceiling* and *robustness*, seeds establish
  *significance*.
* ffmpeg with `libx264` + `libx265` must be on `PATH` for the real-codec anchors
  and for the A3 STE stage (preinstalled on Kaggle).

## Experiment tracking (Comet ML)

Optional, environment-driven, zero behaviour change when off. `train.py` and
`evaluate.py` push live loss curves, per-epoch validation, and the final
BD metrics to a [Comet](https://comet.com) dashboard whenever these variables
are set (`src/tracking.py`):

| Variable | Default | Meaning |
|---|---|---|
| `COMET_API_KEY` | — | **presence enables tracking**; absent → silent no-op |
| `COMET_PROJECT_NAME` | `vcm-preprocessing` | project on Comet |
| `COMET_WORKSPACE` | account default | workspace on Comet |
| `COMET_EXPERIMENT_NAME` | derived from `out_dir` (e.g. `sweep_51-seed_0-mu3`) | experiment name |
| `COMET_MODE` | `online` | `offline` / `disabled` |

`train` writes its experiment key to `<out_dir>/comet_key.txt`, so the later
`evaluate` run **attaches its BD numbers to the same experiment** instead of
spawning a new one. On Kaggle, store the key as a notebook secret
(Add-ons → Secrets, label `COMET_API_KEY`) and `pip install comet_ml`.

## Citations

Full BibTeX-style reference list in [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md#references).
Key sources: Zhao et al. arXiv:2512.15331 (baseline); Lu et al. arXiv:2206.05650 /
TCSVT 2024 (A3 recipe, analyzer-agnostic motivation); Yang et al. TCSVT 2024
(feature-modulation multi-task preprocessor); FiLM (Perez et al. 2018); SFT (Wang
et al. CVPR 2018); DPP (Chadha & Andreopoulos CVPR 2021); Talebi et al. TIP 2021;
J4D arXiv:2606.16185; UG-ICM arXiv:2501.04579; multi-teacher distillation
arXiv:2510.18680; task-driven bit allocation arXiv:1910.07392 & arXiv:2504.02216.
