# RUN DESIGN — Additive retrain in the upgrade_6 harness (§10 of tong-ket-vcm.txt)

Status: DRAFT for sign-off · 2026-08-30 · Target repo: `C:\Users\Wagur1\preprocessing_upgrade_6` (HEAD `6c8e336`)

## 1. Objective and claim thresholds

Train the additive (Zhao-style) preprocessor **inside our canonical harness**, then select its
operating point on **measured real-codec bpp** (gradient-free), and evaluate once on the
canonical 1159-clip protocol.

- Publication target: BD-rate ≤ −8% on BOTH h264 and h265 + gap rule (≥ −0.05 at every QP).
- **Certification rule (pre-registered): to claim ≤ −8% at n=1159, the measured point estimate
  must be ≤ −11% (bootstrap CI95 ≈ ±3 pp).**
- Realistic success criterion (gap ≥ −0.03 is the *effective* rule — pessimistic calibration
  showed a legal −0.05 gap can still price BD-positive).
- Known ceiling arithmetic (rev.py): at ZERO bit cost the current accuracy benefit prices at
  −7.50% (h264) / −8.16% (h265). h264 is the binding side: it needs **both** Δbpp ≤ 0 at the
  selected operating point **and** ~2× the current accuracy benefit. The existence proof is
  Zhao's reported −17.6% h264 at full Kinetics scale; whether 2× exists at 10.8K-clip scale is
  exactly what this run answers.

## 2. Why this run (mechanism, not another variant)

All 6 subtractive axes are closed (R-D neutrality ×7; per-codec law: BD_h265/BD_h264 ≈ 0.03–0.08
for any subtractive filter, so dual −8% is structurally impossible there). Only an
**additive/pre-emphasis** mechanism (gap ≥ 0) can satisfy both codecs simultaneously. The
additive run (best.pt) owns the project's best h265 number (−3.18%) while trained under
worst-case conditions (1,442 clips, ~900 steps, SSF2020 proxy — the least realistic in the
project, alpha=10 λ=0.001 so it never learned about bits). Every learning-failure root cause
identified to date (dead proxy-rate gradient, STE gradient-direction blindness) is **deleted by
construction** here: we train for accuracy only, and choose the bit operating point
gradient-free on measured x264/x265 bpp.

## 3. Method

### 3.1 Architecture — pinned tree, our switches

The 9,795-parameter two-branch additive residual (strict=True pinned by u6_big4/arch.py):

```
spatial_stem 3→16 · spatial_residual.body 16→16→16 · temporal_stem 24→16 (8 frames × 3ch)
fusion.gate 32→16→16 · to_rgb 16→3 · out = x + to_rgb(fused)
```

Fully convolutional, resolution-agnostic, **unconditioned** (no QP input — matches Zhao's one
robust model). The 4 ambiguous wiring switches were never identified (both gates failed; MSE is
non-discriminative — 12 variants within 2.7%) and per §10 they are **ours to choose**:
`act=relu, residual=true, sigmoid gate, lerp fusion (convex blend), center-pick temporal context,
clamp(0,1) on`.

Port: `src/models/additive.py` + `model.arch: additive` branch in `_build_models` + `"arch"`
added to the checkpoint arch-restore tuple in `evaluate()` (the 80724ab fix pattern). Capacity
stays 9,795 params — fixed arch isolates the scale/teacher/proxy levers (widening is arm B,
below, only if arm A saturates).

### 3.2 Objective — Zhao's exact 3-term form

`L = L_Acc + 10·(L_D + 0.001·L_R)` → harness knobs: `lam_task=1, mu=10, beta=0.001,
omega=tau=delta=gamma=0, use_task_mask=false`. The rate term is decorative by design
(α=10 dominates); the proxy's job is the **distortion channel** (x_hat for L_D and L_task), not
its rate gradient. Add `train.clip_grad: 1.0` (the additive run used it; harness lacks it).
Reg-warmup is a no-op here (it ramps only beta/gamma/delta) — correct.

### 3.3 Training proxy — virtual (block-DCT, not SSF2020)

`codec.kind: virtual` with `inter: true` (closed-loop P-frame prediction), `colorspace: yuv420`
(C1), qualities (1,2,3,5,8), C2 in-grid QP→quality mapping `{30:8, 35:5, 40:3, 45:2, 50:1}`,
QP sampled per step. Block DCT + scalar quant + inter prediction + 4:2:0 is the closest match
to x264/x265 quantization noise we have; SSF2020 was the furthest-from-real. Note: differs from
the additive run (3 SSF2020 qualities) — deliberate, C2 is our protocol.

### 3.4 Data — full non-test pool, canonical split hygiene

Same dataset (`rohanmallick/kinetics-train-5per`), same `build_index(CAP_GB=40)` scan → 10,805
clips, then the **mount-independent canonical hash rule** (`_key = <class>/<file>.mp4`):

- **test = md5[0:8]%10==0** → the canonical 1159 (fingerprint `30f083f8520a`) — NEVER touched in training
- **val = md5[0:8]%10==1** (~1,080 clips, deterministic, mirrors the test rule)
- **train = the remaining ~8,644 clips** (6.0× the additive run's 1,442)

New script `scripts/build_train_index.py` emits a harness-format index JSON with labels
(class folder → Kinetics-400 index, same as prepare_3gb). The hash rule is relative-path based,
so the split is stable across Kaggle mounts.

### 3.5 Teachers — A1 panel (our lever, feeds the universality story)

`teachers: [r3d_18, mc3_18], teacher_sampling: sample` (1× analyzer cost/step), eval on
**held-out `r2plus1d_18`** — the canonical eval analyzer. (Zhao used a single analyzer; the
panel is our contribution and a regulariser. Single-teacher is the fallback option at sign-off.)

### 3.6 Geometry & schedule (match the additive run; one T4)

16 frames, stride 2, **frame_size 128** (train + canonical eval both — self-consistent round,
same geometry as the additive run; the analytic-family table at 112 is not directly comparable
and doesn't need to be), analyzer clip_size 112, batch 4 × accum 2 (effective 8, as the run),
lr 1e-4 Adam, cosine, seed 0, val every epoch (`val_qp_mode: all`, `val_max_batches: 5`),
early stop patience 3, `resume: true` wired for session-death recovery (12h Kaggle cap).

Schedule: 2,161 loader-steps/epoch (~1,081 opt steps). fp32 ≈ 2.5 s/step (D9-kernel evidence)
→ ~90 min/epoch → **6 epochs ≈ 9h**; with AMP (optional, smoke-gated) ~10 epochs. That is
6–10× the additive run's ~900 steps. Budget the session at 10h with 2h margin.

## 4. Phases, gates, and cost

| Phase | What | GPU | Wall | Gate to proceed |
|---|---|---|---|---|
| 0 | Port arch + index builder + config + `train.clip_grad` (+AMP flag), unit tests, CPU smoke (10 clips × 50 steps: loss falls, Δ≠0, no NaN) | 0 (local) | ~2h my time | tests green |
| 1 | Train on Kaggle (1 T4 session, repo @port-commit) | ~9–10h | 10h | val loss curve sane; checkpoint saved |
| 2 | Screen + operating-point selection: 120 clips × s∈{0.25, 0.5, 0.75, 1.0} × 5 QP × 2 codecs — measure Δbpp and Δacc; price with `u6_big/budget.py`; pick s* by the pre-registered rule (§5) | ~3h | 3h | Δbpp(s) curve monotone-ish; some s has Δbpp ≤ +2% |
| 3 | Canonical eval at s*: 1159 clips, 6 shards, x264/x265 medium, QP 30–50, per-clip rows, merge + 1000-draw bootstrap CI | ~3.5h (6 slots) | 35 min | — (this produces the verdict) |

**Total ≈ 13–16 GPU-hours of the ~56h pooled quota (10 accounts × 2 slots, 10/10 probe-PASS).**
Straggler/failure handling per auto.py (retry once, abandon after 90 min).

Checkpoint delivery to eval kernels: push `best` checkpoint as a small Kaggle dataset
(`{user}/u6-ckpt-additive`), attach alongside the Kinetics dataset.

## 5. Pre-registered selection rule (Phase 2 → 3)

The model is unconditioned, so the bit operating point is a single global strength
`x_pre(s) = x + s·Δ(x)`. Screen grid (revised 2026-08-31 **after measuring the trained
checkpoint but BEFORE any screen number existed** — see the note below): **s ∈ {0.02, 0.05,
0.1, 0.25, 0.5, 1.0}**. The grid must bracket Δbpp=0 on both sides, and the measurement says
the crossing is BELOW 0.1, not above 1.0: s=0.02 reproduces best.pt's own edit magnitude and
is mandatory; s=1.0 is kept only as the trained reference point. The previous grid
{0.1 … 1.5} is superseded — s=1.5 was insurance for an UNDER-scaled residual, a branch the
measurement kills. Arm count is unchanged (6), so the screen still fits one session per
resolution. On the 120-clip screen (per resolution, 128 AND 224):

1. **Selection is by THRESHOLD, not argmin BD** (pick_s.py, pre-registered):
   `s* = max{ s : gap(s) ≥ −0.03 at every QP, both codecs AND Δbpp(s) ≤ 0 at every QP,
   both codecs }`, with paired bootstrap 5% lower bounds per point ("gap not decidable at
   this n" is printed rather than silently trusted). Rationale: at n≈120 BD spreads 14–16 pp
   (the n=207 lesson: t_stat reads +1.93% and −1.90% at different sample sizes — SIGN FLIP);
   but a one-parameter MONOTONE family is threshold-searchable at n=120, and Δbpp at fixed
   QP contains no analyzer at all. BD is deliberately NOT computed by the selector.
2. If the selector prints "no operating point satisfies the rule": **do NOT conclude the
   mechanism failed** — extend the grid DOWN (s < 0.02) first; the measurement below says the
   failure mode is an over-expensive edit, which lives at small s.
3. Run Phase 3 ONCE per resolution at that resolution's s\*. (A second point s\*±0.25 is
   allowed only if quota ≥ 30h remains.)

**Objective note, superseding the 2026-08-31 mu-cap arithmetic.** The direction of that
correction was right and is now confirmed by measurement, but its numbers were taken from
best.pt and do not describe THIS checkpoint. Measured on the D10 checkpoint (epoch 5 /
step 5395, `u6_big4/edit_size.py`, before Phase 2 was pushed):

| s | residual RMS | edit PSNR | pixels hitting the [0,1] clamp |
|------|------|---------|------|
| 0.10 | 0.053 | 25.6 dB | 3.8% |
| 0.25 | 0.110 | 19.2 dB | 11.5% |
| 0.50 | 0.174 | 15.2 dB | 25.1% |
| 1.00 | 0.245 | **12.2 dB** | 44.6% |

So mu=10 did not merely fail to pin the edit — the edit ran to **19× best.pt's magnitude**
(0.245 vs 0.013 RMS; 12.2 dB vs 37.7 dB), with nearly half of all pixels saturating the
output clamp. At s=1 this is not a prefilter, it is a repaint. best.pt's edit magnitude is
reproduced at **s ≈ 0.025**, which is what fixes the bottom of the grid. Total variation on
smooth synthetic input rises 22% even at s=0.1 (read as direction only — the synthetic
baseline TV is artificially low, so real video will scale less), i.e. the edit ADDS structure
to encode. Expect Δbpp > 0 across most of the grid; the whole selector exists to find where,
or whether, it crosses zero.

## 6. Outcome interpretations (decided in advance)

**Pre-registered expectations (pinned BEFORE any number, per 2026-08-31 review):**

- **Transfer haircut:** the teacher panel [r3d_18, mc3_18] deliberately does NOT contain the
  eval analyzer r2plus1d_18 (the universality claim). Lu et al. measure a 25–30% haircut
  for cross-backbone transfer (−22.0% on-teacher → −16.4/−15.7% transferred). Therefore, to
  measure −11% (the certification threshold) on the held-out analyzer, the on-teacher number
  must be ≈ −15%, i.e. **Δacc ≈ +0.065, not +0.05**. Evaluate the screen against this bar.
- **The untested variable is CAPACITY, not steps:** D10 multiplies data 6× (1,442 → ~8.6k
  clips) and steps ~7× (901 → ~6,483), but the model stays at 9,795 parameters vs Lu et al.'s
  9.42M (962×). At this size, 6.5k steps most likely converge — so a flat D10 result does
  NOT close the additive branch; it opens the capacity question (arm B / width becomes the
  next experiment, not a retraction of the mechanism).

Verdicts:

- **Measured BD ≤ −11% both codecs, gap pass** → the method paper is alive; replicate seed; add
  tracking task; move to publication checklist.
- **−4…−8% territory** → scale ceiling is real at 10.8K clips; report honestly; the neutrality
  paper gains the additive-scale datapoint; decide then whether a 2nd seed / arm B is justified.
- **≈ 0 ± 3% or worse** → R-D neutrality holds even for the additive family at this scale —
  **read as a capacity datapoint, not a mechanism refutation** (see above); the
  neutrality/boundary-condition paper gets its capstone — write it.
- **Δbpp(s=1) far worse than the additive run's (+14.8%→+4%)** → the virtual-codec distortion
  channel pushed the edit the wrong way. **This diagnostic has ALREADY fired, before Phase 2
  returned any bpp**: the edit statistics in §5 put D10 at 12.2 dB against best.pt's 37.7 dB.
  So a large positive Δbpp at s=1 is now the EXPECTED reading, not a surprise, and it is not
  by itself evidence against the mechanism — it is evidence that the useful operating point is
  s ≈ 0.02–0.05. What would count against the mechanism is Δbpp > 0 at *every* grid point, or
  gap collapsing to ≤ −0.03 at the s where Δbpp finally turns negative (i.e. the accuracy gain
  is NOT separable from its bit cost — the one genuinely open question this screen answers).

## 7. Risks & mitigations

- **Early residual blow-up** (no `max_relative_edit` envelope, faithful to Zhao): mu=10 L_D
  active from step 0 + `clip_grad 1.0` + CPU smoke gate.
- **AMP numerics** with the DCT proxy: flag `train.amp`, default **off**; enable only if the
  50-step fp32-vs-AMP loss curves agree; fp32 fallback = 6 epochs (already 6–10× the run).
- **Session death at hour 9**: resume=true + last.pt checkpointing; auto.py retry.
- **Val-cost creep** (all-QP val): val_max_batches=5 caps it at ~100 forwards/epoch.
- **Non-comparability with the 112-clip analytic table**: accepted by design (self-consistent
  128 round; anchor vs prep at identical settings).

## 8. Non-goals (the forbidden list holds)

No STE/real-forward 4th attempt; no best.pt 224 push; no pricing on the unidentified rebuild;
no r144; no per-codec QP sub-grids; no low-QP branch; no new resample sizes; no rate-allocation/
QP-field; no tstack; no `robust_transfer_ste.yaml`. Single seed first (a 2nd seed only on a
promising Phase 3, before any paper claim). Tracking task and frame-224 stay out of this round.

## 9. Decisions requested at sign-off

1. Approve the plan + ~13–16 GPU-h spend (of ~56h pooled)?
2. Teachers: panel [r3d_18, mc3_18] + held-out r2plus1d_18 (recommended) or single r3d_18?
3. AMP: implement + smoke-gate (recommended), or skip (fp32, 6 epochs)?
4. Arm B (width-32, same topology) queued only if arm A shows saturation — OK to leave out now?
