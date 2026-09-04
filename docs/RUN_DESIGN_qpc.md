# Round (b) — QP-Conditioned Additive Edit

**Status: pre-registered 2026-09-04 evening, before any run. Commit: TBD (this
doc + implementation land together). Trainer account: TBD (post quota-refresh
2026-09-05 07:00 local). Config: `configs/additive_qpc.yaml` (fork of
additive_ar; the kappa=10 lineage config stays untouched).**

## The measured failure this round attacks

kappa=10 (project best) is *untargeted in rate* — structural reason #1 of the
additive-valid post-mortem: the edit costs bits at EVERY QP
(Δbpp +13.1/+11.5% @QP30 → +6.0/+1.8% @QP50) but earns accuracy only at the
heavy end (+0.029…+0.061 gaps, all five QPs positive but small at QP30). At
QP30 we pay full price for a gap the codec did not need to preserve. Every
dead amplitude lever (omega, kappa_t, mu=3) tried to make the edit CHEAPER or
BIGGER; none tried to make it *conditional on where in the R-D curve the
codec is*.

## Mechanism

Give the existing 9,795-param additive editor a rate condition (FiLM, zero-init)
so it can spend edit amplitude where it pays:

- **Low QP (light compression):** the codec preserves the source well; the
  accuracy gap is small. The optimal edit is SMALL — stop paying bits for a
  gap the codec itself already covers.
- **High QP (heavy compression):** the codec destroys analyzer-relevant
  detail; the gap is large. The edit should concentrate here — the same
  accuracy gain per bit costs fewer bits per unit gap when the codec is
  quantising hard (its own rate cost is low there).

The BD integral then improves on BOTH ends: fewer added bits at the high-rate
end of the anchor curve, larger effective gap at the low-rate end. This is a
bit-ALLOCATION lever, not an amplitude lever — the axis the failure analysis
says was never tried ("all tried bit-levers were GLOBAL penalties; the
targeting axis — QP-conditional / spatial — never measured").

Zero-init FiLM makes conditioned == unconditioned at step 0: training can only
DEVIATE from the kappa=10 optimum if the loss gradient says so. Downside is
bounded by the incumbent's own behavior; the risk is it never deviates (a
null result, see expectations).

## Design decisions (each with a reason)

1. **FiLM on the shared trunk, not per-branch.** One `FiLM(cond_dim=1, ch=16)`
   between fusion and to_rgb: conditions the single 16-ch representation both
   branches feed. Per-branch FiLM doubles the new parameters for a
   condition signal that is a scalar per clip — one shared trunk FiLM is the
   minimal, cheapest conditioning point.
2. **cond = qp_norm(qp) in [0,1]** (1 = QP51 = heavy compression), same
   `_rate_cond`/`_qp_norm` the engine already builds for the U-Net path — no
   engine changes needed.
3. **gate input: unmodified cond, zero-init FiLM.** The condition enters
   additively inside FiLM's MLP (Linear(1→16)→LeakyReLU→Linear(16→32)); the
   second Linear is zero-init so γ=β=0 at init → exact identity vs the
   unconditioned model → the *unconditioned* 9,795-key checkpoints
   strict-load into the base keys (state_dict gains 4 new keys:
   `film.0.{weight,bias}`, `film.2.{weight,bias}`; verified by unit test).
4. **Param budget: 9,795 → 10,371 (+576, +5.9%).** Negligible capacity
   change; capacity was never the binding constraint (the 224 retrain showed
   the LOSS BALANCE shifts, not capacity, at 3× pixels).
5. **No saliency mask this round (spatial targeting deferred).** One new
   variable at a time — the standing project rule. QP-conditioning is the
   temporal/rate axis; spatial gating (use_task_mask + masked_tv) is a
   SEPARATE round (b2) only if (b) lands center/upside. Registered follow-up
   condition, not a promise.
6. **Eval feeds the TRUE QP of each encode.** The canonical eval loop
   (mk_eval_kernel cell) already knows the QP per arm — it passes
   `cond=_rate_cond(qp_norm(qp))` to the conditioned model. QP30 arm gets
   cond 0.303, QP50 gets 0.645 (qp_ref [20,51]).
7. **Train config = kappa=10 lineage except the arch key.** 16ep, batch 8,
   128, seed 0, teacher panel [r3d_18, mc3_18] sampled, mu=10, kappa=10,
   beta=0.001, clip_grad 1.0, cosine, patience 4. The ONLY deltas:
   `model.arch: additive_cond`, `out_dir: outputs/additive_qpc`.

## Pre-registered expectations (written before launch)

Comparators (kappa=10 lineage, 2-shard): h264 −3.42 [−5.88,−0.88] /
rep1 −2.52 [−4.92,+0.16]; h265 −2.63 [−4.49,−0.79] / rep1 −2.33 [−4.14,−0.57].
Read with ±1pp re-run noise (rep1 finding).

- **Center (~35%): h264 −2.5…−3.5 / h265 −2.3…−3.2** — net wash vs kappa=10:
  the condition learns something small but the incumbent is already
  near its family's ceiling.
- **Upside (~15%): h264 −4…−5 / h265 −3…−4.2** — the edit genuinely
  concentrates at high QP: Δbpp@QP30 drops toward ≤+10% while QP45/50 gaps
  hold or grow. Mechanism signature (checkable in the store rows): the
  per-QP Δbpp curve flattens downward at the light end.
- **Downside (~50%): h264 −1.5…−2.5 / h265 −1.5…−2.5** — FiLM trains as a
  global amplitude nuisance (batch-level noise), or the model cannot infer
  "the codec will be heavy" from a scalar well enough within 16ep, and the
  extra 336 params just add variance. 5/5 prior accuracy attempts landed
  downside; this prior dominates by count.

## Gates before any eval spend (0 GPU, local)

Gates run on `dl_qpc/.../preprocessor.pth` with `edit_size_qpc.py` /
`added_hf_qpc.py` (new; feed cond explicitly, default cond=QP45-normalized
0.613 — the operating regime the edit is FOR):

1. **Regime gate:** added HF @s=0.25 ∈ −5…+30% at BOTH cond=QP30 (0.303) and
   cond=QP50 (0.645). The wide band mirrors mu=3's registration: a
   conditioned edit may legitimately grow.
2. **Conditionability gate (the NEW instrument):** spread(added HF) across
   cond ∈ {0.303, 0.645} at s=1.0 — reported, no pass/fail (no prior for
   what a "useful" spread is; this run DEFINES the prior for round b2).
3. **No-blow-up gate:** RMS @s=1.0 < **0.14** (incumbent mu=10 twin reads
   0.1202, mu=3 read 0.1289 on the same script — threshold set ABOVE the
   incumbent's own readout this time; Q3(iii) hygiene fix).
4. **Identity at init:** untrained conditioned model with any cond is exact
   identity (zero-init FiLM + zero-init to_rgb) — unit test, not a gate.

**Gate discipline:** regime gate (1) must PASS on both conds, else NO eval.
Gates 2-3 are reported numbers; RMS > 0.14 = automatic no-eval.

## Guards (registered before the run)

- **(i) 1D-axis discipline:** this is the QP-axis point at kappa=10/mu=10
  fixed. A downside landing does NOT close "targeting" as a direction (spatial
  gating remains unmeasured); an upside landing does not mean the (QP ×
  spatial) interaction is additive.
- **(ii) FiLM-utilization audit before quoting any mechanism:** report
  ‖γ‖,‖β‖ at the two eval conds from the checkpoint — if the network
  effectively zeroed the FiLM, the run is a re-trained kappa=10 (null
  conditionability), and the honest read is "conditioning available but not
  used at this budget", not "targeting failed".
- **(iii) Same-instrument comparisons only:** all comparators from the same
  edit_size/added_hf script family; the mu=3 letter-fail lesson.
- **(iv) One run, one variable:** any hyperparameter drift beyond the
  registered deltas voids the comparison — the run config diff must be
  exactly {arch, out_dir}.

## Ops plan (morning, post quota-refresh 07:00 local 2026-09-05)

1. `python -X utf8 mk_train_kernel.py <account> <commit> "train.epochs=16"
   additive_qpc.yaml` (trainer account chosen from the refreshed pool;
   htran123456-by-default unless its quota is still drained).
2. Watcher: background shell + auto_resume.log pattern has died silently 6×
   on this machine (Rule 3). Use the durable-cron backstop ONLY (hourly
   status check → relaunch watcher if kernel mid-run; on COMPLETE → gates).
3. On COMPLETE: pull → local gates (above) → report to user → USER decides
   eval spend (1-shard direction screen first, as with mu=3).
