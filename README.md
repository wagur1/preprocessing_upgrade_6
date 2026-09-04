# preprocessing_upgrade_6

**Tiền xử lý video phổ quát, không phụ thuộc analyzer, cho VCM (Video Coding for
Machines), đặt trước một codec chuẩn *đóng băng* (x264 / x265).**

> ## English summary — status first
>
> A universal, analyzer-agnostic video preprocessor for Video Coding for Machines,
> placed in front of a **frozen** standard codec (x264/x265). Only the small
> preprocessor is trained: no bitstream change, no decoder change, no per-CTU QP
> map. The optimised quantity is the preprocessor's gain on the *same* codec,
> `BD-Rate(prep+x264 vs x264)` and `BD-Rate(prep+x265 vs x265)`, on the accuracy
> axis (Kinetics-400 top-1), with a **held-out** analyzer never used in training.
>
> **Honest state: the −8% publication target is not met, and the number-chasing
> track is CLOSED on measured evidence — the project's main line is now the
> reposition battery (universality + statistical hardening), with the −2.5%
> headline positioned as the first rigorous measurement of this niche.**
> Target was BD ≤ −8% on *both* codecs with the accuracy-gap rule passing
> (`prep − anchor ≥ −0.05` at every QP).
>
> | | measured, canonical protocol (n=1159, held-out `r2plus1d_18`, QP 30–50) |
> |---|---|
> | best variant passing the gap rule | `kappa=10 @16ep` **−3.42%** h264 [CI95 −5.88, −0.88], **−2.63%** h265 [−4.49, −0.79] |
> | replication (seed 1) | h264 **−2.52%** [−4.92, +0.16] (CI spans 0), h265 **−2.33%** [−4.14, −0.57] (significant HELD) |
> | quoting rule (cross-review Q1) | report **min/max/n**, never a mean; **h265 leads the headline** (significant in both runs, flat across the kappa sweep) |
> | certification arithmetic | CI ≈ ±3 pp at n=1159, so *claiming* ≤ −8% requires **measuring ≤ −11%** |
>
> **Four findings that are the actual contribution, negative or not:**
>
> 1. **R-D neutrality, stated as an exchange rate — and where it breaks.** The
>    6-epoch checkpoint beat the encoder's own QP knob in 1 of 10 cells; the
>    16-epoch kappa=10 checkpoint **wins 8/8 comparable cells at 1.07×–2.86×**.
>    The two separable levers: a long schedule supplies accuracy (10/10 cells)
>    but raises bits 2.5× faster (BD worse alone), while the RPP-style adaptive
>    DCT penalty (arXiv:2301.10455) restrains bits at held accuracy — the only
>    penalty here that reshapes the residual's SPECTRUM rather than its
>    amplitude. Protocol flips the sign: the same screen re-scored on the
>    training teacher instead of the held-out analyzer turns BD positive →
>    negative, which is why the literature's −12.3…−19.6% (per-backbone,
>    on-teacher, no CI) is not comparable to this protocol.
> 2. **A measured, lineage-corrected ceiling.** Holding measured accuracy fixed
>    and scaling the edit's bit cost to k× its real value (`u6_big4/ceiling.py`,
>    overlap window k-independent by construction): the kappa=10 lineage prices
>    at **−11.89% (h264) / −8.03% (h265)** at zero bit cost (k=0.25 already
>    gives −9.79/−6.69). Two earlier readings are retracted: the −19.76/−10.49
>    numbers came from the broken-temporal-branch checkpoint on a 113-clip
>    screen, and the published −9.47% h265 was the *f4gr0* lineage's ceiling,
>    not kappa=10's. The "gap ×1.55 ⇒ −8% on both codecs" stack extrapolation
>    is likewise RETIRED (cross-review C5: one-point linear extrapolation in a
>    family whose only accuracy experiment ran backwards). Current honest
>    statement: −8% needs k ≈ 0.45 (h264) to k ≈ 0.0 (h265) — the h265 margin
>    is 0.03 pp under the most generous possible assumption, and no measured
>    lever has entered the low-k region. The −8% target stays "alive" only in
>    this arithmetic sense.
> 3. **A complete, pre-registered falsification map (2026-09-02 → 09-04).**
>    Every remaining accuracy/breadth lever was run with bands registered
>    before launch, and every one landed in its registered downside or worse:
>    `omega=1.0` feature distillation −1.55/−1.56 (teacher-overfit signature,
>    as Liu et al. 1910.09185 predicted); `mu=3` freed-edit −2.12/−0.71
>    (1-shard; bits rose faster than gap — Δbpp@QP30 +15.9/+13.9 vs kappa=10's
>    +14.2/+11.5; real-clip edit RMS grew only +18%, so the mu axis is
>    *dominated*, not saturated); `kappa_t=50` −1.85/−2.50 (aimed at the wrong
>    cell — the 0-GPU Parseval-exact spectrum audit retracted the 3-D DCT
>    round's premise BEFORE the run, commit 5dc7737); geometry @224 +0.01/+5.04
>    (the analyzer's fixed 112 resize is a Nyquist wall for the 128-trained
>    edit); `add224` retrain gate-failed (smoothing regime). The one axis never
>    measured is *allocation* (QP-conditional / spatial), pre-registered as
>    round (b) QP-conditioned FiLM (`docs/RUN_DESIGN_qpc.md`, commit 15283bd,
>    cross-review-audited, NOT yet run) and a hard δ-cap (round c) — both
>    queued BEHIND the reposition battery.
> 4. **The reposition battery is the main line** (checklist:
>    `Desktop/vcm-reposition-checklist-2026-09-04.md`): 6-backbone universality
>    matrix (have r2plus1d/r3d/mc3; add i3d, SlowFast, MViT2), tracking via
>    DiMP/PrDiMP, fine-tuned-analyzer baseline, VMAF row, third seed, s-grid
>    {0.15, 0.25, 0.35} — all eval-only on the existing checkpoint except the
>    seed run. Literature anchors per cross-review C8: the only published
>    pre-only datapoint (Google's Sandwiched Compression, arXiv:2402.05887) is
>    10–15% on HUMAN metrics (LPIPS) with 7.85M params, and the whole sandwich
>    is ~5% on HEVC/PSNR — our −2.5% is *consistent with* that band; it is
>    never quotable as "the niche's band" for machine metrics.
>
> **Latest (2026-09-04):** the number-chasing track closed with the mu=3
> round (registered downside landed: 1-shard h264 −2.12 / h265 −0.71, bits
> rose faster than gap at held gaps — the pre-registered failure family, not a
> blow-up: gap rule PASS at every QP). A full-protocol replication of kappa=10
> with seed 1 (`rep1`) exposed ~1 pp of re-run variance at the eval/codec
> level (edit-level twin: same best epoch, RMS 0.0452 vs 0.0467), which
> downgrades the h264 claim to "marginal-to-significant" and anchors the
> min/max/n quoting rule. Round (b) — QP-conditioned additive edit, one
> zero-init FiLM on the shared trunk (10,371 params, `71146d9`, design revised
> per cross-review C7 at `15283bd`) — is implemented, unit-tested (suite
> 80/80), pre-registered, and waiting on quota behind the reposition battery.
> All results, per-QP data, and the k-scaling ceiling live in
> `Desktop/vcm-results-2026-09-04.xlsx`; the narrative deck for non-experts is
> `Desktop/vcm-bao-cao-tong-ket-2026-09-04.pptx` (Vietnamese). *(The gamma-null
> report this paragraph replaces — TV penalty decorative at 0.00–0.04% of the
> objective, `gamma_res` mapping onto amplitude — remains valid history and is
> kept in the Vietnamese body below.)*
>
> **Cross-validated against an independent implementation.** A teammate reimplemented
> the same paper's preprocessor from scratch (`munnn01/proxy_v3`, read at source level
> with permission). The two agree on everything the checkpoint cannot pin — module tree
> 3→16 / 16→16→16 / 24→16 / 32→16→16 / 16→3, sigmoid gate with convex blend
> `gate*sp + (1−gate)*tp`, zero-init output projection, clamp to [0,1] — which is what
> made the single disagreement worth acting on: our temporal branch stacked the clip's
> centre 8 frames ONCE and broadcast the result to every frame, so it contributed a
> per-clip constant and the residual could not follow motion. It is now a causal window
> ending at the current frame. Independently, their rate penalty carries a temporal
> difference our total variation lacked — so our bit-cost proxy had been blind to the
> inter-frame residual, which is where a video codec spends most of its bits. Both are
> fixed and pinned by tests; parameter count is unchanged (9,795). Deliberately NOT
> adopted: their straight-through codec bridge is byte-identical to ours, whose gradient
> is `∂bpp_proxy/∂θ` exactly and has now failed in five places; and their
> outside-the-analyzer-crop rate lever does not apply here, because this analyzer
> resizes to 112 rather than centre-cropping, so there is no invisible region.
>
> **Reproduction:** canonical split fingerprint `30f083f8520a` (train 8636 / val
> 1010 / test 1159, `rohanmallick/kinetics-train-5per`), real x264/x265 via
> ffmpeg at preset medium, QP {30,35,40,45,50}. `pytest` → 80 tests.
> **A number missing any one validity condition is a diagnostic, not a result** —
> that boundary is enforced throughout this document.
>
> *The body below is in Vietnamese.* Design and pre-registration:
> `docs/RUN_DESIGN_additive.md`.

Chỉ một mạng tiền xử lý nhỏ được huấn luyện. Codec và mọi mô hình thị giác phía
sau ("analyzer") giữ nguyên, đúng chuẩn — không sửa bitstream, không sửa decoder,
không bản đồ QP theo CTU. Bộ tiền xử lý sửa pixel *trước khi* encode, sao cho ở
cùng một bitrate, máy vẫn nhìn thấy thứ nó cần.

```
                    (được train)            (ĐÓNG BĂNG)              (panel ĐÓNG BĂNG)
 video x ─► Preprocessor θ ─► x_pre ─► Codec chuẩn ─► x̂ ─► Analyzer(s) ─► nhãn cho máy
            U-Net + FiLM(rate)         x264 / x265              r3d_18 / mc3_18 / …
            + SFT(motion)              (train qua proxy yuv420) + 1 analyzer HELD-OUT khi eval
            + D1 GATE: edit *= (1−M)   M = saliency task của analyzer đóng băng
            hoặc D4.5: ramp theo QP    (kết quả không-cần-train)
            hoặc D10: residual cộng    (họ additive, đã định giá 31/08)
```

Đại lượng được tối ưu là **độ lợi của bộ tiền xử lý trên *cùng* một codec**:
`BD-Rate(prep+x265 vs x265)` và `BD-Rate(prep+x264 vs x264)` — âm nghĩa là bộ
tiền xử lý giúp máy đạt cùng độ chính xác với ít bit hơn. (So sánh chéo codec vẫn
được báo cáo, nhưng QP không so được giữa hai codec nên chỉ mang tính tham chiếu.)

---

## Trạng thái dự án (2026-09-04) — đọc mục này trước

**Headline trung thực (luật min/max/n theo phán quyết cross-review Q1):**
`kappa=10 @16ep, s=0.25` — h264 **−3.42% [−5.88, −0.88]** / rep1 seed=1 **−2.52%
[−4.92, +0.16]** (CI chạm 0); h265 **−2.63% [−4.49, −0.79]** / rep1 **−2.33%
[−4.14, −0.57]** (cả 2 significant). **h265 dẫn headline** — significant ở cả 2
lần chạy, phẳng qua toàn bộ kappa sweep (không exposure winner's-curse). Nhiễu
lặp lại ~1pp sống ở tầng EVAL/codec, không phải train (twin edit-level: cùng best
epoch, RMS 0.0452 vs 0.0467) → mọi so sánh dùng dung sai ±1pp.

**Bản đồ falsification khép lại 2026-09-02 → 09-04 (mọi round đăng ký band TRƯỚC
khi chạy, mọi round rơi đúng downside đã ghi):**

| round | BD h264 / h265 | cơ chế chết |
|---|---|---|
| omega=1.0 (chưng cất đặc trưng) | −1.55 / −1.56 | teacher-overfit — đúng signature Liu 1910.09185 dự báo |
| mu=3 (giải phóng edit) | −2.12* / −0.71* | bit tăng nhanh hơn gap (Δbpp@QP30 +15.9/+13.9); edit thật chỉ +18% RMS → trục mu bị *dominated*, không đóng |
| kappa_t=50 (phạt thời gian) | −1.85 / −2.50 | nhắm sai cell — audit phổ Parseval 0-GPU retract premise TRƯỚC run (`5dc7737`) |
| geometry @224 input | +0.01* / +5.04* | analyzer resize về crop 112 = tường Nyquist với edit train ở 128 |
| add224 retrain | gate-fail | regime mượt hóa (vùng kappa=20/30) — không tốn eval |
| 3-D DCT round | chưa chạy — retract | audit năng lượng: escape ở spatial-MID temporal-AC, sai cell |

(* = 1-shard 554 clip, đọc hướng.) **Kết luận: track đuổi số ĐÓNG.** Giá trị còn
lại = phép đo đầu tiên của niche + bản đồ cơ chế; **main line là battery
reposition** (`Desktop/vcm-reposition-checklist-2026-09-04.md`): ma trận 6
backbone, tracking DiMP/PrDiMP, baseline fine-tuned, VMAF, seed-3, s-grid.

**Trần lý thuyết đã đính chính theo lineage (audit C5, `u6_big4/c5c_audit.py`):**
kappa=10 → **−11.89% h264 / −8.03% h265** ở k=0 (miễn phí bit hoàn toàn); k=0.25
→ −9.79/−6.69. Số cũ −9.47 (h265) là của dòng f4gr0, không phải kappa=10. Cửa sổ
overlap k-independent by construction (đã kiểm bằng code) — artifact rd_neutral
không thể xảy ra ở đây. Mục tiêu −8% "còn sống" duy nhất theo nghĩa số học: cần
k ≈ 0.45 (h264) tới k ≈ 0.0 (h265); không lever đo được nào vào được vùng k thấp.
Claim "gap ×1.55 ⇒ −8% cả hai codec" đã RETIRE (C5: ngoại suy 1 điểm).

**Round (b) — QP-conditioned additive edit** (`docs/RUN_DESIGN_qpc.md`, commit
`15283bd`): thêm MỘT zero-init FiLM trên trunk chung của editor (9.795 → 10.371
params; best.pt strict-load phần base — có test chốt); premise đã bị Claude bắt
sai lineage trong cross-review C7 và sửa bằng audit 0-GPU: gap/Δbpp lệch
**4.1× (h264) tới 10.2× (h265)** giữa QP30 và QP tốt nhất — target reallocation
tồn tại thật. Đăng ký band + gate hành vi (cond QP30 vs QP50 phải lệch > ±3% twin
noise) + escalation b' (BD-weighted QP sampling chỉ nếu null-with-FiLM-used) +
δ-cap là round (c). Cả hai xếp SAU battery.

**Số liệu đầy đủ:** `Desktop/vcm-results-2026-09-04.xlsx` (12 run × 2 codec, CI,
per-QP, k-scaling, gates — mọi số từ merge.py trên store Kaggle thật).

## Trạng thái dự án (2026-08-31) — lịch sử, giữ nguyên để đối chiếu

Mọi con số phía dưới trong lịch sử D-series được đo trên **207 clip**. Chúng đã
được đo lại trên **1159 clip** với khoảng tin cậy ở mức clip, và kết luận đã đổi.

**Số duy nhất còn đứng vững theo protocol chuẩn** (1159 clip, fingerprint
`30f083f8520a`, analyzer held-out `r2plus1d_18`, QP 30–50, preset medium):

| biến thể | cơ chế | BD h264 [CI95] | BD h265 | luật gap |
|---|---|---|---|---|
| `t_base` | ramp blur + saliency mask tĩnh | **−3.41%** [−5.67, −1.24] | −0.26% | PASS (−0.015) |
| `x_base` | như trên, mask từ analyzer **held-out** | −2.65% [−4.64, −0.49] | **−1.43%** | PASS (−0.011) |
| `f_s4` | strength phẳng s=0.4 | −1.14% [−3.44, +0.98] | −0.21% | PASS (−0.029) |
| `tdup6` | giữ frame theo thời gian, mỗi frame thứ 6 | **−8.18%** [−11.30, −5.22] | −0.23% | **FAIL (−0.176)** |
| `g224` | cờ encoder theo từng codec, `frame_size` 224 | −0.65% [−3.81, +2.32] | −0.84% | FAIL (−0.072) |
| `r96` | resample vòng về 96² | +8.54% [+5.57, +11.60] | +8.69% | FAIL (−0.144) |
| `lo_s7` | s=0.7 phẳng trên lưới QP20–40 | +9.28% [+2.70, +15.06] | +5.62% | FAIL (−0.066) |
| `f4gr0` s=0.25 | residual cộng, nhánh thời gian causal (`f4e5f05`) | −1.31% [−3.53, +1.16] | **−2.79% [−4.61, −0.96]** | PASS, gap dương mọi QP |
| **`kappa10-16ep` s=0.25** | **như trên + hạng DCT `kappa=10`, 16 epoch** (`70cd452`) | **−3.42% [−5.88, −0.88]** | **−2.63% [−4.49, −0.79]** | **PASS, gap dương mọi QP** |

**Luật gap** (quy tắc tuyển chọn xuyên suốt): chênh lệch độ chính xác
`prep − anchor` phải **≥ −0.05 tại mọi QP** trên cả hai codec. Vi phạm là bị loại,
BD-rate đẹp cỡ nào cũng không cứu.

**Chưa biến thể nào đạt mục tiêu công bố** (BD ≤ −8% trên *cả hai* codec, gap PASS).
Cái vượt −8% duy nhất — `tdup6` — hỏng gap gấp 3.5 lần luật.

**KẾT QUẢ TỐT NHẤT HIỆN TẠI — `kappa10-16ep` (2026-09-02): biến thể ĐẦU TIÊN có ý
nghĩa thống kê trên CẢ HAI codec.** h264 −3.42% [−5.88, −0.88] (P(BD<0)=0.996) và
h265 −2.63% [−4.49, −0.79] (P=1.000), cả hai CI không chứa 0, gap **dương ở mọi QP**
trên cả hai. Trước đó `t_base` chỉ significant ở h264 (h265 −0.26%) và `f4gr0` chỉ ở
h265 (h264 CI trùm 0). Vẫn **chưa đạt** mục tiêu −8% (cần đo ≤ −11%).

**Hai lever tách rời và cộng dồn — đây là điều làm nó chạy:**
1. **Lịch học dài cấp ACCURACY, không cấp BD.** Một biến, 6→20 epoch: accuracy tăng
   ở **10/10** ô, nhưng bit tăng nhanh **gấp 2.5 lần** accuracy, nên BD **xấu đi**
   (h264 −1.31% → −0.99%). Đừng dùng số epoch như một lever BD.
2. **`kappa` (adaptive DCT kiểu RPP, arXiv:2301.10455) kìm BIT.** Δbpp@QP30 h264:
   18.3% (6ep) → 23.8% (20ep) → **14.2%** (kappa=10 @16ep) mà accuracy vẫn giữ. Đây
   là hạng phạt **duy nhất** trong repo nặn lại *phổ* của residual thay vì bóp *biên
   độ*: RMS 0.05075 → 0.04386 trong khi năng lượng HF thêm vào giảm +24.2% → +6.9%.

**Chỗ rò đã đo, và là bước tiếp theo:** `kappa` là DCT theo khối **trên từng frame**,
thuần không gian — model lách sang trục thời gian (`TVt/RMS` 0.4931 → 0.6964 khi lịch
dài ra). Y hệt cách `gamma_res` bị lách (t-share 37.2% → 42.8%). **Chặn một trục thì
model dồn chi phí sang trục chưa chặn — đo được hai lần, trên hai hạng phạt độc lập.**
Bản sửa là DCT khối **3-D** thay vì từng frame.

**Mọi run learned của dự án đều bị cắt lịch học.** `epochs: 6` sinh ra từ một comment
ước `~90 min/epoch` khi thực tế là **23**. Best epoch đo được: additive **17/20**,
kappa=10 **14/16**, và D1/D2 **vẫn đang cải thiện ở epoch 13** khi Kaggle kill ở giới
hạn 12h/session. Nên mọi kết luận "học thất bại" trong sổ trục đều rút ra ở khoảng
1/3 lịch học khả dụng.

**Điểm vận hành đã chốt: s=0.25.** `s=0.02` cho −0.10%/−0.52% (CI trùm 0) — gần
identity, BD → 0 đúng như phải vậy, và đó là sanity check cho kết quả ở s=0.25.

**Hai cái lần đầu, do `f4gr0` (2026-09-01).** (1) CI của h265 **không chứa 0**
([−4.61, −0.96], P(BD<0)=1.000) — số âm có ý nghĩa thống kê đầu tiên trên h265 kèm
gap PASS. (2) Gap **dương thật ở mọi QP trên cả hai codec** (+0.015…+0.064 h264,
+0.028…+0.079 h265), không chỉ trên sàn −0.05. Và **tỉ lệ per-codec bị đảo**:
h265/h264 = 2.13, trong khi mọi cơ chế *trừ bớt* cho 0.03–0.08 — **h265 không còn
là codec chặn của họ additive, h264 mới là** (CI của h264 vẫn chứa 0). Đây là một
**đánh đổi**, không phải cải thiện toàn diện: h264 xấu hơn `t_base`. Hướng là
*enhancement* (Δbpp dương ở mọi QP, +2…+18%), cùng phía với Zhao-reference và
`film_deeper3d` — khác hai cái đó, nó âm có ý nghĩa, nên nó **hạn định** phát biểu
trung hoà R-D chứ không phủ định: −2.79% nằm ở rìa âm của dải ±3%.

**Số học để chứng nhận:** với n=1159, CI ≈ ±3 điểm phần trăm, nên muốn *claim*
≤ −8% thì phải **đo được ≤ −11%**. Đây là ngưỡng thật, không phải −8%.

### Mô hình độ dốc: đổi accuracy ra bit

Quan hệ thực nghiệm giữa gap độ chính xác và BD-rate, fit trên các vòng đã đo:

```
BD ≈ exp(−slope · Δacc) − 1        slope = 2.45 (h264), 2.9 (h265 @224)
```

| Δacc cần có | BD h264 kỳ vọng | BD h265 kỳ vọng |
|---|---|---|
| +0.00 | 0% | 0% |
| +0.02 | −4.8% | −5.6% |
| +0.03 | −7.1% | −8.3% |
| +0.05 | −11.5% | −13.5% |

Đọc theo chiều ngược lại: muốn đo được −11% (ngưỡng chứng nhận), cần
**Δacc ≈ +0.048 trên h264** — tức bộ tiền xử lý phải làm analyzer chính xác **hơn**
video gốc gần 5 điểm, không chỉ "không làm hỏng". Mô hình này *thiên vị* các Δacc
lớn (dạng exp), nên với Δacc > 0.05 phải đo thật thay vì ngoại suy.

### Vòng mới nhất

**D10 — họ additive** (`src/models/additive.py`, 9.795 tham số, theo Zhao et al.):
`x_pre = x + s · to_rgb(fused)`, hai nhánh không-gian/thời-gian trộn bằng cổng
sigmoid. Đây là cơ chế duy nhất có thể cho gap ≥ 0 — mọi trục *trừ bớt* (blur,
drop frame, resample) đều đã đóng vì trung hoà R-D (xem dưới).

Lần chạy đầu **VOID** vì lỗi hiệu chuẩn proxy (xem "Bẫy hạ tầng #1"). Đã sửa ở
commit `2101bf1` — cổng đã PASS ở đây là *proxy-sanity* (identity qua proxy đã
sửa: top-1 0.009→0.336–0.850) và demo `virtual_codec`, **không phải**
`scripts/smoke_additive.py`, cái này hỏng từ lúc zero-init `to_rgb` cho tới khi
được sửa. Train lại đã **xong** (6 epoch, 2h24m, best epoch 5) và đã định giá:
gap dương thật ở QP45/50 nhưng BD dương ở 12/12 arm — xem
`Desktop/vcm-additive-2026-08-31.txt`.

**Đối chiếu với một implementation độc lập (31/08).** Teammate dựng lại preprocessor
của cùng paper từ đầu (`munnn01/proxy_v3`, đọc ở cấp source, có phép). Hai bên trùng
nhau ở **mọi** thứ mà checkpoint không chốt được — cây module `3→16 / 16→16→16 /
24→16 / 32→16→16 / 16→3`, sigmoid gate với convex blend `gate*sp + (1−gate)*tp`,
zero-init `to_rgb`, clamp [0,1] — chính vì thế **một** điểm bất đồng mới đáng tin.
Điểm đó: nhánh thời gian của ta xếp 8 frame **giữa** clip **một lần** rồi broadcast
ra mọi frame, nên nó đóng góp một **hằng số theo clip** và residual không thể theo
chuyển động (với clip 16 frame, edit áp lên frame 0 tính từ frame 4–11). Nay là cửa
sổ **causal** kết thúc tại frame hiện tại. Việc này chốt 1 trong 4 switch wiring mà
`docs/RUN_DESIGN_additive.md §2` ghi là không xác định được từ checkpoint.

Độc lập với đó, hạng rate của họ có **hiệu theo thời gian** mà `total_variation` của
ta thiếu — nên proxy chi phí bit của ta **mù với residual liên khung**, đúng chỗ codec
video tiêu phần lớn bit. Cả hai đã sửa và có test chốt; số tham số không đổi (9.795)
nên `best.pt` vẫn strict-load.

**Cố ý không lấy:** STE bridge của họ (`standard_codec.py:664-666`) byte-identical với
`src/models/ste_codec.py:78-79` của ta — `∂bpp/∂θ = ∂bpp_proxy/∂θ` chính xác, đã chết
ở 5 chỗ. Đòn "rate ngoài crop analyzer" không áp dụng được ở đây: analyzer của ta
**resize** về 112 (`src/tasks/action_recognition.py:81-91`) chứ không center-crop, nên
không tồn tại vùng vô hình. Còn proxy distill từ codec thật (`train_proxy.py`: L1 với
recon thật đã cache + smooth-L1 với bpp đo được) là đường **duy nhất chưa thử** —
nhưng cache của họ chỉ chứa codec chạy trên clip **gốc** trong khi proxy đóng băng,
nên cần giải quyết distribution shift trước khi port.

Ba kết luận của vòng này, vì chúng đổi cách đọc cả dự án:

**(1) Phép biên tập thua chính núm QP của encoder — nhưng CHỈ ở checkpoint
6-epoch; đã bị đảo ở checkpoint hội tụ.** BD-rate thực chất chỉ hỏi một câu: bộ
tiền xử lý mua accuracy có rẻ hơn cách đơn giản là hạ QP không? Đo accuracy trên
mỗi %bit (`u6_big4/accperbit.py`, cùng 113 clip screen), **checkpoint 6-epoch**:
phép biên tập thắng núm QP ở **1/10 ô** duy nhất (h265 QP50, 1.33×, mà ở đó
accuracy chỉ 0.07→0.22 — task đã vỡ); ở h264 QP45 nó **kém núm 1.26×**, ở h264
QP30 nó **âm** (−0.00061 so với +0.00181 của núm) — trả +73% bit để *mất*
accuracy.

**Đính chính (2026-09-02):** con số 1/10 là thuộc tính của một checkpoint bị cắt
schedule, không phải của cơ chế. `kappa=10 @16ep` (project best, h264 −3.42%
[−5.88, −0.88] / h265 −2.63% [−4.49, −0.79], gap dương ở mọi QP) **thắng núm QP ở
8/8 ô so sánh được**, tỉ giá 1.07×–2.86×. Phát biểu đúng bây giờ: trung hoà R-D
là tỉ giá *mặc định*, và nó bị phá khi và chỉ khi phần biên tập được huấn luyện
đủ dài **và** chi phí bit bị một penalty phổ giữ lại — hai lever tách rời và cộng
tính (schedule cấp accuracy nhưng một mình làm BD xấu vì bit tăng nhanh hơn 2.5×;
`kappa` giữ bit lại: Δbpp@QP30 h264 18.3% → 23.8% → **14.2%**).

**(2) Accuracy KHÔNG còn là điểm nghẽn — chi phí bit mới là.** Giữ accuracy đo
được cố định rồi ép chi phí bit về k lần mức thật (`u6_big4/ceiling.py`): phần
lợi model **đã** đạt định giá **−19.76% h264 / −10.49% h265** nếu residual miễn
phí. h264 vượt xa mốc −11% cần để chứng nhận −8%. Nhưng trần h265 là −10.49%
**ngay cả ở chi phí bằng 0**, nên mục tiêu "−8% trên *cả hai* codec" là
**bất khả thi về số học** ở mức accuracy hiện tại (h265 thực tế: −3…−5%). Đây là
phát biểu mạnh hơn "chưa biến thể nào đạt mục tiêu" ở đầu file — nó đổi cả cuộc
thảo luận về mục tiêu công bố.

**(3) Protocol đảo dấu, và điều đó giải thích literature.** Cùng screen, sửa
**một dòng** config (`eval.held_out_backbone: r2plus1d_18` → teacher `r3d_18`):
BD h264 ở s nhỏ **đảo sang âm** (−5.40 / −7.60 / −1.88% tại s=0.02/0.05/0.10, so
với +4.22 / +5.36 / +0.57% khi held-out). Cơ chế: phân bố gap theo QP đảo —
teacher 8/10 ô không-âm vs held-out 3/10 — tức analyzer overfit đậu đúng vào vùng
bit rẻ. **Suy ra: con số −12.3…−19.6% của Zhao là số *theo từng backbone*
(on-teacher); thước của repo này (cả hai codec, analyzer held-out) NGHIÊM HƠN
chính claim của literature.** Cảnh báo giữ nguyên: **không được trích −7.60%** —
n=113, không đơn điệu theo s, boot5% dứt khoát ở 1/5 ô; phát hiện là **việc đảo
dấu có hệ thống**, không phải độ lớn.

Hướng thật sự còn mở **không phải** "chạy lại D10" mà là **objective biết đến chi
phí bit**: `configs/additive_ar.yaml` ship `gamma: 0.0` (TV) và `delta: 0.0`, với
`beta: 0.001` mà chính comment gọi là decorative — model **chưa bao giờ** được
yêu cầu tạo residual *rẻ*, trong khi `edit_size.py` đo TV tăng **1.39×**. Núm cần
xoay là `gamma` (TV), **không phải `omega`** (`omega` là feature distillation —
`src/losses.py:3-4` là nguồn đúng về tên các hạng); và `gamma` chứ không phải
`beta`, vì `src/losses.py:32-38` gọi TV là "the lever for transfer to x264/x265
(unlike `beta*bpp`, which only reduces the *proxy* codec's bits)" và bpp của proxy
không đáng tin làm mục tiêu tối ưu (Bẫy #1). Rủi ro phải nói trước: phần lợi
accuracy có thể **chính là** phần tần số cao, hạ TV thì mất luôn.

---

## Cơ chế đã tìm ra: trung hoà R-D

Các biến thể ở trên không thất bại vì những lý do rời rạc. Chúng thất bại vì **một**
lý do, và nó đo được theo từng QP:

> **Tại đúng độ chính xác mà clip đã tiền xử lý đạt được, anchor trần cũng chỉ cần
> xấp xỉ số bit đó.**

Hỏi chính đường cong `(độ chính xác → log bpp)` của anchor xem một mức chính xác
đáng giá bao nhiêu bit, thì mọi khoản tiết kiệm của prefilter đều đúng bằng thứ nó
đã phá. Một cái blur tiết kiệm 11% bitrate và mất 0.03 độ chính xác, trên đường
cong của anchor, **không phân biệt được với việc chỉ đơn giản tăng QP**.

Đã xác nhận trên **năm cơ chế độc lập**: blur không gian, cổng saliency, drop frame
theo thời gian, phân bổ rate không gian (trường QP theo block), và — lần đầu trên
thứ *không phải* prefilter — **cấu hình phía encoder** (`g224`: `-tune ssim`,
`psy-rd=0:psy-rdoq=0`). Tắt RDO tâm thị giác tiết kiệm bit *nhiều hơn* dự đoán và
vẫn trả về BD ≈ 0.

Hai hệ quả phải nói thẳng, vì chúng tốn GPU-giờ mới học được:

- **Đo riêng bitrate không phải bằng chứng có lợi thế.** `lo_s7` tiết kiệm đều
  −9…−12% ở mọi QP và trả về **+9.28%**, vì đường cong anchor ở QP20 dốc đến mức
  một gap −0.029 định giá thành +73% bitrate-tương-đương. Ngân sách accuracy để hoà
  vốn rất nhỏ ở QP thấp và nới ra khi QP tăng.
- **Sàng lọc 3 clip cục bộ phải được hiệu chuẩn, và hiệu chuẩn không phổ quát.**
  `r96` cho thấy sàng lọc *lạc quan* 0.60×; `g224` cho thấy nó **bi quan 1.63×**.
  Khác biệt nằm ở độ bão hoà của anchor, không ở cỡ mẫu.

### Kết quả hình học: tại sao mọi số h265 đều ≈ 0

h265 ra gần 0 ở gần như mọi vòng, và đó không phải codec cứng đầu. Ở 112², h265 chỉ
trải **×3.38 bitrate trên toàn lưới QP30–50** (h264 trải ×6.19), với bước theo QP
suy giảm −36.9/−30.0/−22.7/−13.5% — bước giảm đơn điệu là dấu vết của một **sàn
overhead cố định**. Đoạn QP45→50 gần như dựng đứng trong mặt phẳng rate–accuracy,
nên tích phân BD trên đó bị điều kiện xấu (ill-conditioned): mọi phần trăm tiết
kiệm đều đo trên một đường cong hầu như không dịch chuyển.

Ở 224² cùng phép đo cho **×5.02 với h265** (h264 ×6.92) với bước gần như không đổi,
và độ chính xác QP50 của h265 tăng từ 0.115 lên 0.250. **Vấn đề là hình học, không
phải lưới QP** — điều này cũng khai tử luôn đề xuất "lưới QP riêng cho từng codec".

### Sổ trục (axis ledger)

| trục | trạng thái | bằng chứng |
|---|---|---|
| blur không gian / cổng saliency | **đóng** | trung hoà R-D; cả họ đơn điệu theo strength, cực trị là identity tiếp cận từ dưới |
| drop frame theo thời gian | **đóng** | `tdup6` BD −8.18% nhưng gap −0.176 |
| phân bổ rate không gian (trường QP theo block) | **đóng** | ở bitrate *khớp nhau*, không gì thắng được việc để encoder tự phân bổ |
| chroma | **đóng** | bỏ chroma **hoàn toàn** chỉ tiết kiệm 3–6% bitstream |
| cấu hình encoder theo codec | **đóng** | `g224` BD ≈ 0, gap h264 FAIL |
| hình học 224 + bỏ octave trên cùng | **đóng** | `g224` / `r96`; analyzer resize về 112 nên octave trên vô hình, nhưng bỏ nó vẫn không thắng |
| **residual cộng (additive, gap ≥ 0)** | **MỞ — trục duy nhất có số âm chứng nhận trên CẢ HAI codec** | `kappa10-16ep` s=0.25: h264 **−3.42% [−5.88, −0.88]** / rep1 −2.52, h265 **−2.63% [−4.49, −0.79]** / rep1 −2.33, gap dương mọi QP, 1159 clip. Chưa đạt −8% (trần k=0: −11.89/−8.03, audit C5). Các lever phụ đã đóng: omega (teacher-overfit), mu=3 (bit-dominated), kappa_t (sai cell), 3-D DCT (retract pre-run `5dc7737`), @224 (Nyquist-112). Bước tiếp theo dòng chính: **battery reposition**; cơ chế còn duy nhất chưa đo: **allocation** — round (b) QP-FiLM (`15283bd`, pre-registered) + δ-cap (round c) |

Một kết quả âm trong dòng phân bổ rate đáng giữ lại: **bản đồ saliency có mang
thông tin thật**. Bảo vệ *nửa sai* số block (cùng số lượng, mask lật ngược) làm mất
0.32 / 0.24 độ chính xác so với bảo vệ nửa đúng. Nên "mask không có tác dụng" là
tính chất của *việc blur*, không phải của bản đồ.

---

## Bẫy hạ tầng (đọc trước khi tin bất kỳ con số nào)

Ba mục dưới đây mỗi cái đã từng huỷ ít nhất một vòng chạy. Chúng ở trong README vì
đọc lại rẻ hơn tái phát hiện.

### #1 — Bước lượng tử hoá của proxy tính theo đơn vị [0,1], KHÔNG phải [0,255]

`VirtualCodec` mã hoá các plane trong **[0,1]**. Với DCT 8×8 trực chuẩn, hệ số DC
nằm ở `8·mean ≈ 4` và gần như mọi hệ số AC đều < 0.5. Nên `step = 1.0` **xoá sạch
toàn bộ AC** và đặt giá trị trung bình của block lên lưới `step/8` — khoảng 8 mức
xám, chỉ còn thấy block. Số mức xám ngụ ý = `1 / (step/8)`.

`configs/additive_ar.yaml` từng ship `step_coarse: 3.0 / step_fine: 1.0` — những
con số **hợp lý với JPEG trong thang 8-bit** và **thô gấp 255 lần** ở đây. Hệ quả,
đo tại 128², 16 frame:

| | QP30 | QP35 | QP40 | QP45 | QP50 |
|---|---|---|---|---|---|
| x264 thật | 31.44 | 28.85 | 26.32 | 23.70 | 21.52 dB |
| x265 thật | 31.14 | 28.48 | 25.81 | 23.26 | 21.00 dB |
| proxy, step 3.0/1.0 (**hỏng**) | 19.16 | 15.19 | 15.71 | 12.59 | **9.71** dB |
| proxy, step 0.25/0.03 (**đúng**) | 35.86 | 32.04 | 28.73 | 27.53 | 25.40 dB |

Setting **mịn nhất** của proxy hỏng (19.16 dB) còn tệ hơn quality **thô nhất** của
x264 mà nó phải đại diện (21.52 dB). Nên analyzer ở mức **đoán bừa trên từng frame
nó từng thấy** trong suốt quá trình train: `L_task` chạy CE 8.484 → 8.038, chưa bao
giờ xuống dưới `ln(400) = 5.991`, và **76% mức giảm loss là `mu·L_D` fit vào một
target đã bị phá**. Residual thu được là nhiễu bị MSE nắn theo rác.

Xác nhận trực tiếp: cho identity (`s = 0`, không sửa gì) đi qua chính proxy đó,
113 clip Kinetics thật, `r2plus1d_18` held-out:

| | clean | QP30 | QP35 | QP40 | QP45 | QP50 |
|---|---|---|---|---|---|---|
| proxy **hỏng**, top-1 | 0.894 | 0.018 | 0.009 | 0.009 | 0.027 | 0.009 |
| proxy **đã sửa**, top-1 | 0.894 | **0.850** | **0.743** | **0.602** | **0.478** | **0.336** |
| proxy đã sửa, CE | 0.492 | 0.570 | 1.003 | 1.889 | 2.629 | 3.835 |

0.009–0.027 trên 400 lớp **chính là mức đoán bừa** (113 clip ⇒ 1 clip = 0.00885).

**Quy tắc rút ra:** hiệu chuẩn proxy khả vi theo **distortion so với codec thật**,
**không bao giờ theo bpp**. Chính việc chạy theo bpp "trông thực tế" đã làm step bị
thô: rate model Gaussian không tham số đếm thiếu bit, và với `beta = 0.001` thì số
hạng rate chỉ mang tính trang trí — bpp cao hơn x264 5–20× là **đúng và không liên
quan**. Đo target bằng `u6_big4/proxy_target.py`.

### #2 — Assertion đơn điệu là một lỗ gác, không phải một cái gác

Lỗi #1 lọt qua self-check vì assertion duy nhất trên phần hiệu chuẩn là tính đơn
điệu (`mse[0] > mse[-1]`) — **mà một tấm hình bị phá hoàn toàn vẫn thoả**. Bất kỳ
assertion chỉ-đơn-điệu trên một núm điều chỉnh độ trung thực đều là lỗ gác.

`virtual_codec._demo()` giờ gác bằng **sàn tuyệt đối**: quality mịn nhất > 24 dB
(phải thắng quality tệ nhất mà nó đại diện), thô nhất > 15 dB. Chạy
`python -m src.models.virtual_codec` để kiểm tra.

### #3 — STE / forward bằng codec thật đã chết 3 lần, đừng đề xuất lần thứ 4

Cùng một ý tưởng, ba repo, ba lần thất bại. STE sửa **giá trị** mà loss nhìn thấy,
nhưng backward vẫn là của proxy — nên nó **không sửa hướng gradient**. Hình học
sai vẫn cho hướng sai. Xem `docs/RUN_DESIGN_additive.md`.

### #4 — Cỡ mẫu test

3 GB / 207 clip làm BD-rate trên h264 trải rộng thêm 14–16 điểm phần trăm. **~600
clip là sàn**; 1159 clip gần như miễn phí so với 600. Mọi bảng xếp hạng giữa các
biến thể đo trên 207 clip đều là fit nhiễu — đó chính xác là điều đã xảy ra với
D1–D9.

### #5 — Chia train/test và chia shard theo MD5 của đường dẫn clip, không theo vị trí

`build_index` shuffle một list mà thứ tự đến từ `os.walk`, thứ này **không ổn định
giữa các máy**. Chia theo vị trí sẽ âm thầm cho các shard chồng lấn nhau. Mọi shard
đều assert cùng fingerprint test `30f083f8520a`.

---

## Các đóng góp hạ tầng

### A1 — Bộ tiền xử lý phổ quát, không phụ thuộc analyzer (đóng góp chính)
`src/tasks/multi_teacher.py`, `src/tasks/base.py::build_analyzer`

Thay vì một analyzer đóng băng duy nhất, bộ tiền xử lý được train trước một **panel
teacher đóng băng** (`task.teachers`, ví dụ `r3d_18` + `mc3_18`). Mỗi step hoặc lấy
trung bình task loss trên toàn panel (`mean`), hoặc **lấy mẫu một teacher**
(`sample` — một cách regularize kiểu stochastic-multi-teacher, ngăn edit chuyên biệt
hoá vào một mạng). Distillation đặc trưng đi theo teacher *đang hoạt động* để giữ
nhất quán trong step.

Tuyên bố về tổng quát hoá sau đó được đo trên một **analyzer held-out chưa bao giờ
có trong panel** (`eval.held_out_backbone`, ví dụ `r2plus1d_18`). Các công trình
tiền xử lý trước codec chuẩn (Lu et al. arXiv:2206.05650) chỉ cho thấy transfer
*hẹp* trong cùng họ; các công trình chứng minh được transfer rộng (**UG-ICM**
arXiv:2501.04579, **All-in-One Transfer** arXiv:2504.12997) đều **train lại codec**.
"Tiền xử lý phổ quát, chứng minh trên analyzer held-out rộng, *với codec chuẩn giữ
nguyên đóng băng*" là khe hở mà repo này nhắm tới.

**Giới hạn đo được của chính A1, phải đọc kèm:** panel teacher **không** ngăn được
chuyên biệt hoá ở quy mô này. Vòng D10 chạy panel 2 teacher, vậy mà phân bố gap
theo QP ở s nhỏ vẫn **đảo dấu có hệ thống** giữa teacher và held-out (8/10 vs 3/10
ô không-âm, xem "Vòng mới nhất" điểm 3). Nên A1 hiện là **một cách đặt vấn đề kèm
phép đo cho thấy nó chỉ transfer một phần**, chưa phải một đóng góp đã chứng minh.
Đó chính là lý do mọi con số trong file này đều báo song song hai protocol.

### A2 — Bản đồ tầm quan trọng theo không gian
`src/models/task_mask.py`, `src/losses.py`

Bản đồ **gradient-saliency** của task loss theo input, `m = |∂L_task/∂x|`, tính bằng
một backward phụ qua teacher đóng băng rồi **detach**. Nó *tái phân bổ theo không
gian* các penalty về edit (`delta`) và total-variation (`gamma`) theo `1 − m`, nên bộ
tiền xử lý làm mượt và ngừng tiêu bit ở **nền**, chừa lại vật thể mà máy cần. Đây là
bản tương tự khả vi, ở miền pixel, của phân bổ bit theo task — nhưng không chạm vào
encoder.

### A3 — Hiệu chuẩn với codec thật (proxy → thật)
`src/models/ste_codec.py`, `src/models/virtual_codec.py`

1. **Straight-through codec thật.** `STECodec` chạy x264/x265 *thật* ở forward và
   vay gradient của proxy khả vi ở backward:
   `x̂ = x_proxy + (x_real − x_proxy).detach()`, tương tự cho bpp.
   **Kết quả của repo này: chết 3 lần** — xem Bẫy hạ tầng #3.
2. **Anneal lượng tử mềm→cứng.** Proxy block-DCT anneal quantiser từ nhiễu đều cộng
   (mềm) sang straight-through hard rounding (`codec.anneal: 1.0`), để proxy kết thúc
   ở đúng quantiser cứng của codec thật.

### C1 — Proxy trong không gian màu yuv420 (khe hở màu mà STE không đóng được)
`src/models/color.py`, `src/models/virtual_codec.py`

x264/x265 **không bao giờ** mã hoá RGB: chúng đổi sang BT.601 YCbCr, hạ mẫu chroma
2×2 (**ở mọi QP**, độc lập với bitrate), và lượng tử chroma thô hơn (offset QP chroma
của H.26x ≈ +6 QP ở QP50). Một proxy RGB do đó **tính thiếu giá** cho các edit nặng
chroma: bộ tiền xử lý cứ tiêu ngân sách vào chi tiết chroma mà codec thật phá miễn
phí. Proxy 5.1 tái tạo toàn bộ hình học đó dưới dạng op khả vi:

```
RGB ─► BT.601 YCbCr ─► hạ mẫu chroma 2×2 ─► DCT+quant theo plane (chroma thô hơn ×2)
  ─► upsample chroma ─► YCbCr → RGB
```

`colorspace: rgb` giữ đường cũ cho bảng ablation.

### C2 — Protocol QP trong lưới
`configs/*.yaml` (`train.qp_list = eval.qp_list = [30, 35, 40, 45, 50]`)

Điều kiện hoá rate bằng FiLM chỉ được train trên đúng những QP nó thấy; QP eval ngoài
lưới train là ngoại suy (triệu chứng nhìn thấy được ở upgrade-3: độ chính xác
`prep+h265` *sụp* ở QP50). 5.1 train đúng trên lưới eval với năm quality proxy phân
biệt — đóng khe hở này với chi phí bằng 0.

---

## Lịch sử: chuỗi D-series đã dẫn tới đây

> ⚠️ **Toàn bộ số trong mục này đo trên 207 clip và đã bị bảng 1159-clip ở đầu
> README thay thế.** Giữ lại để biết *cách* đã đi tới kết luận, không phải để trích
> dẫn. Thứ tự xếp hạng giữa các biến thể ở đây là fit nhiễu.

**Cái gì đã kết thúc upgrade-5.1.** Hai lần chạy 3-seed đầu tiên ra **âm**
(`prep+h264 +2.1%`, `prep+h265 +3.2%`). Một chiến dịch 11 biến thể trọng số loss sau
đó vẽ ra toàn bộ không gian khả đạt: biến thể tốt nhất (`distill2`, ω=2) vẫn cho
QP30-gap −0.213, vẫn hỏng luật −0.05; và **loss chính xác của Zhao không transfer
sang hạ tầng này** (gap −0.298, BD +87%/+60%). Hai cơ chế được xác lập: (i) task
cross-entropy **chết về mức đoán bừa** dưới thiệt hại của proxy ở QP nặng, nên phần
lớn thời gian train *không có gradient bảo toàn độ chính xác*; (ii) không còn tín
hiệu task sống, các áp lực còn lại lái edit về "rẻ cho codec, không đọc được cho
analyzer". Mọi cách sửa phía loss đều đã thử và thất bại ⇒ thất bại là **cấu trúc**.

| chặng | làm gì | kết quả |
|---|---|---|
| **D1/D2** | cổng saliency cấu trúc `x_pre = x + (1−M)·edit` | cổng chạy đúng như thiết kế (vùng bảo vệ là identity bit-exact, QP30-gap hồi 1/3) nhưng residual U-Net học ra **nhiễu tốn bit**: bitrate x264/x265 thật **tăng ~5%** ở mọi QP |
| **D3/D3.1/D4** | tham số hoá edit thành blend về blur (`edit_kind=smooth`), 5 cấu hình độc lập kể cả STE | **cả 5 đều học ra đúng identity** (s≈0, BD 0.00%) |
| **D4.5** | bỏ hẳn việc học: ramp cố định theo QP | **kết quả dương đầu tiên**, BD −1.99%/−2.34%, 9/9 quan sát seed×codec đều âm |
| **D5–D7** | lưới 17 cấu hình quanh ramp60 | phong cảnh lõm, blur heuristic **bão hoà ở ≈−3%**; đẩy strength mạnh hơn làm BD *xấu đi* +2…+5% |
| **D8** | thay rate model chết bằng prior factorized có train | mở đường, nhưng bị bảng 1159-clip vượt qua |
| **D10** | residual **cộng** kiểu Zhao (trục duy nhất còn sống) | lần 1 VOID vì Bẫy #1. Chạy lại `2101bf1` **hợp lệ**: 6 epoch/2h24m, gap dương thật ở QP45/50 nhưng BD dương 12/12 arm. Phân tích trần: accuracy **đã đủ** (−19.76% h264 nếu residual miễn phí), điểm nghẽn là **chi phí bit** (+62%) |

**Nguyên nhân gốc của D3/D4** (đọc thẳng từ `VirtualCodec._quant_rate`): rate model
là công thức Gaussian-power không tham số `R = ½·log2(1 + 12·E[y²])`. Khi nền đã bị
blur, hệ số DCT rơi xuống dưới bước lượng tử và `∂R/∂s → 0` — **gradient rate chết
đúng ở nơi làm mượt sẽ có lợi**. Gradient CE luôn sống sau đó đẩy s→0 không ai cản.

**Ghi chú lịch sử đáng giữ:** lần chạy D3 đầu tiên "pass" với −12…−20% bpp — một lỗi
eval âm thầm nạp trọng số `smooth` vào kiến trúc residual; sửa ở `80724ab`,
`evaluate()` giờ phục hồi siêu tham số kiến trúc từ config lưu trong mỗi checkpoint.
Bài học: **soi forward pass trước khi tin những con số đẹp.**

### Bậc thang đóng góp (cùng dữ liệu 3 GB, cùng protocol — đã bị thay thế)

| phương pháp | BD h264 / h265 | chi phí |
|---|---|---|
| 5.1 learned (tốt nhất trong 11 biến thể) | +2.1% / +3.2% | train đầy đủ |
| u6 learned + gate (tốt nhất) | +47.7% / +32.5% | train đầy đủ |
| u6 ramp60 cố định | −2.0% / −2.3% | **không cần train** |
| Zhao et al. reproduce @ 3 GB (400 video) | +0.77% / −3.18% | train đầy đủ |
| Zhao et al. báo cáo (full Kinetics, A100, theo từng backbone) | −12.3…−19.6% | train theo từng backbone |

> Dòng Zhao reproduce (+0.77%/−3.18%) đo trên **n=400, trên teacher `r3d_18` chứ
> không phải analyzer held-out**, và checkpoint đó **không tái tạo được** (MSE gate
> không phân biệt được kiến trúc; 144 biến thể decode đều fail). **Không dùng để
> claim.**

---

## Cấu trúc repo

```
src/
  models/
    preprocessor.py    U-Net + FiLM(rate) + SFT(motion) sửa pixel (được train)
                        + cổng saliency D1/D2 (gate, gate_area)
                        + tham số hoá smooth D3 (edit_kind=smooth)
    additive.py        D10 residual cộng hai nhánh kiểu Zhao (9.795 tham số)
    virtual_codec.py   proxy block-DCT khả vi (+ anneal mềm→cứng A3, yuv420 C1)
    entropy_codec.py   D8 rate model Laplacian factorized-prior có train
    codec.py           proxy học được CompressAI (codec train thay thế)
    ste_codec.py       A3 wrapper straight-through cho codec thật
    task_mask.py       A2 bản đồ gradient-saliency + TV có mask
    color.py           C1 BT.601 RGB↔YCbCr + hạ/nâng mẫu chroma 4:2:0
  tasks/
    base.py            interface TaskAnalyzer + build_task / build_analyzer
    multi_teacher.py   A1 panel teacher đóng băng
    action_recognition.py  phân loại video Kinetics-400 (r3d_18/mc3_18/r2plus1d_18)
    tracking.py, siamfc.py, pytracking_adapter.py  task tracking GOT-10k
  codecs/standard.py   x264/x265 thật qua ffmpeg (bpp coded trung thực)
  losses.py            L_task + ω·L_distill + β·bpp + τ·L_temp (+ δ,γ có mask)
                       (+ γ_res·TV(x_pre−x)) (+ κ·L_dct) (+ μ·L_D)
                       TV gồm cả hiệu thời gian; L_dct = adaptive DCT kiểu RPP
  metrics/bd_rate.py   BD-Rate/BD-accuracy Bjøntegaard trên đường cong rate–accuracy
  engine.py            vòng train / eval, điều kiện hoá rate, BD-Rate 6 pipeline
configs/
  universal_action_recognition.yaml   config chính A1+A2+A3 (+ khoá D-series)
  additive_ar.yaml                   vòng D10 additive (ĐỌC comment về step!)
  action_recognition.yaml, tracking.yaml   baseline một analyzer
docs/
  RUN_DESIGN_additive.md   thiết kế + pre-registration + §5.2 quy nguyên nhân D10 + §5.3 kết quả chạy lại
  bao_cao_preprocessing.md báo cáo tiếng Việt của vòng trước
  MODEL.md, IMPROVEMENTS.md, KAGGLE.md
kaggle/   notebook Kaggle chạy ngay + launcher
tests/    62 unit test, gồm self-check gate/smooth/entropy/additive/residual-TV
```

Phần lớn module không tầm thường có self-check `__main__` (8 module: `virtual_codec`,
`ste_codec`, `task_mask`, `color`, `bd_rate`, `multi_teacher`, `prepare_3gb`,
`prepare_got10k`). `additive.py` **không** có — check của nó nằm ở
`tests/test_additive.py` (pytest-collectable, 7 test):

```bash
python -m src.models.virtual_codec    # gồm cả cái gác PSNR tuyệt đối (Bẫy #2)
python -X utf8 tests/test_additive.py  # additive: 9.795 tham số, identity lúc init, đại số strength
python -m src.tasks.multi_teacher
python -m src.models.task_mask
python -m src.metrics.bd_rate
```

---

## Chạy thử

### Một phát trên Kaggle

Attach `rohanmallick/kinetics-train-5per`, bật GPU + Internet, chạy một cell:

```bash
%%bash
set -euo pipefail
cd /kaggle/working
if [ -d preprocessing_upgrade_6/.git ]; then
  git -C preprocessing_upgrade_6 pull --ff-only
else
  git clone -q https://github.com/wagur1/preprocessing_upgrade_6.git
fi
cd preprocessing_upgrade_6
bash kaggle/run.sh
```

`kaggle/run.sh` tự phát hiện thư mục Kinetics đã mount và dựng lại các index cũ
không chứa split `test` độc lập.

### Chạy tay

```bash
pip install -r requirements.txt    # torch, torchvision, compressai, opencv, ffmpeg trên PATH

# 1) dựng index dữ liệu (chia theo MD5 — xem Bẫy #5)
python scripts/build_train_index.py --help

# 2) train qua proxy khả vi
python train.py --config configs/additive_ar.yaml

# 3) eval trên analyzer HELD-OUT, anchor x264/x265 thật, BD-Rate
python evaluate.py --config configs/additive_ar.yaml \
    --ckpt outputs/additive_ar/checkpoints/preprocessor.pth \
    eval.split=test eval.held_out_backbone=r2plus1d_18
```

Checkpoint chỉ lưu **trọng số bộ tiền xử lý**, nên nó độc lập với codec: eval cùng
một checkpoint trên codec khác bằng cách đổi `codec.kind` / danh sách anchor. Kết quả
(`results.json`, `curves.csv`, `rate_accuracy.png`, `qualitative.png`) nằm ở
`outputs/eval/`.

Với `eval.per_sequence: true` (mặc định cho action recognition), evaluator còn ghi
`sequence_points.csv`, `sequence_bd_rate.csv`, `sequence_bd_rate.json` — năm điểm QP
và BD-Rate cùng-codec cho từng video held-out. `top1` theo video được giữ làm chẩn
đoán nhị phân; vì một giá trị nhị phân không dựng được đường cong 5 điểm hữu ích cho
phần lớn video đơn lẻ, phần fit BD theo sequence dùng `target_prob` — xác suất mà
analyzer đóng băng gán cho lớp ground-truth. Con số headline vẫn là BD-Rate trên
top-1 toàn tập.

---

## Protocol đánh giá & metric

`src/metrics/bd_rate.py` tính **BD-Rate với độ chính xác của máy làm trục chất
lượng** (top-1 cho recognition, AUC success-plot cho tracking) thay cho PSNR, tích
phân `log(rate)` trên vùng độ chính xác chồng lấn (Bjøntegaard). Mặc định, eval vẽ
`prep+{x264,x265}` so với `{x264,x265}` trần và báo:

* **`bd_prep_gain`** — độ lợi cùng-codec (`prep+x265 vs x265`, …). **Đây là claim thật.**
* `bd_vs_anchor` — chéo codec (chỉ tham chiếu; QP không so được giữa hai codec).

BD-Rate âm = ít bit hơn ở cùng độ chính xác.

**Ba điều kiện để một con số được coi là hợp lệ trong repo này:**

1. Đo trên **test set chuẩn**: 1159 clip, fingerprint `30f083f8520a`, không bao giờ
   chạm tới lúc train.
2. Đo trên **analyzer held-out** (`r2plus1d_18`), không phải teacher.
3. **Luật gap PASS**: `prep − anchor ≥ −0.05` ở mọi QP, trên cả hai codec.

Thiếu bất kỳ điều nào thì con số là chẩn đoán, không phải kết quả.

---

## Dữ liệu

* **Kinetics-400** (Kay et al., 2017) — action recognition; các video ResNet đóng
  băng của torchvision giữ đúng thứ tự 400 lớp chuẩn. Dataset Kaggle
  `rohanmallick/kinetics-train-5per` (35.8 GB) → 8636 train / 1010 val / 1159 test.
* **GOT-10k** (Huang et al., TPAMI 2021) — tracking một vật thể; AUC success-plot /
  AO / SR. Tracker mặc định là SiamFC tự chứa; KYS/DiMP/ATOM/PrDiMP đúng như bài báo
  chạy qua `pytracking` (`scripts/install_pytracking.sh`).

## Ghi chú về tái lập

* Repo này là **hạ tầng nghiên cứu**, không phải một bộ số đã đóng băng.
* ffmpeg có `libx264` + `libx265` phải nằm trên `PATH` cho các anchor codec thật
  (Kaggle đã cài sẵn).
* BD-rate trên tập nhỏ dao động ±3–4% mỗi seed. Ở 1159 clip, hãy báo **CI ở mức
  clip** bằng bootstrap trên các dòng per-clip (`kaggle/report_ci.py`), *không* dùng
  độ tản của 3 seed — CI clip-level là thanh sai số trung thực cho "phương pháp này
  trên tập test này".
* Mỗi shard eval ghi cả dòng per-clip (bpp, hit top-1, xác suất GT) và các bộ tích
  luỹ cộng dồn, nên phần merge dựng được CI 95%.

## Theo dõi thí nghiệm (Comet ML)

Tuỳ chọn, điều khiển bằng biến môi trường, không đổi hành vi khi tắt.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `COMET_API_KEY` | — | **có mặt là bật tracking**; không có → no-op im lặng |
| `COMET_PROJECT_NAME` | `vcm-preprocessing` | project trên Comet |
| `COMET_WORKSPACE` | mặc định của account | workspace |
| `COMET_EXPERIMENT_NAME` | suy ra từ `out_dir` | tên thí nghiệm |
| `COMET_MODE` | `online` | `offline` / `disabled` |

`train` ghi experiment key vào `<out_dir>/comet_key.txt` để lần `evaluate` sau
**gắn số BD vào cùng thí nghiệm** thay vì sinh cái mới.

---

## Những việc KHÔNG làm nữa

Danh sách này tồn tại vì mỗi dòng đã tốn GPU-giờ để đóng lại.

- Lần thứ 4 với STE / forward bằng codec thật trên họ *trừ bớt* (Bẫy #3).
- Định giá bất kỳ vòng nào trên một checkpoint không tái tạo được (`best.pt` cũ).
- "Sửa" bpp của proxy bằng cách làm thô step lượng tử (Bẫy #1 — đúng là sai lầm đã
  huỷ D10).
- Assertion chỉ-đơn-điệu trên núm độ trung thực (Bẫy #2).
- Lưới QP riêng cho từng codec, nhánh QP thấp, thêm cỡ resample, `r144` — hình học
  ở 224 đã trả lời hết.
- Phân bổ rate / trường QP, `tstack`, bỏ chroma — đã đóng.
- Chọn metric cho vừa kết quả; đọc kết luận từ một shard merge chưa xong.
- Mở rộng lưới strength xuống dưới khi gap đơn điệu âm — scale nhiễu về 0 chỉ trả về
  identity, không tạo ra tín hiệu.

## Trích dẫn

Danh sách tham khảo đầy đủ ở [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md#references).
Nguồn chính: Zhao et al. arXiv:2512.15331 (baseline); Lu et al. arXiv:2206.05650 /
TCSVT 2024 (công thức A3, động lực analyzer-agnostic); Yang et al. TCSVT 2024
(tiền xử lý đa nhiệm điều biến đặc trưng); FiLM (Perez et al. 2018); SFT (Wang et al.
CVPR 2018); DPP (Chadha & Andreopoulos CVPR 2021); Talebi et al. TIP 2021;
J4D arXiv:2606.16185; UG-ICM arXiv:2501.04579; distillation đa teacher
arXiv:2510.18680; phân bổ bit theo task arXiv:1910.07392 & arXiv:2504.02216;
Sandwiched Compression arXiv:2402.05887.
