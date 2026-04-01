## Decision: Branch Portfolio — MLP Adoption + Calibration + Domain-Shift Mitigation

### Current State
- **Best run**: run-0003-0001 at val ROC-AUC 0.6652 (class-coverage-expansion-probe)
- **Lane leader**: run-0002-0003 MLP probe head at val ROC-AUC 0.6801
- **Train-val gap**: 0.992 → 0.665, indicating severe overfitting or domain shift
- **Padded_cmap**: 0.062 (weak segment-level localization)
- **Prior fusion**: 0.662, barely different from raw val — prior adds negligible lift

### Decision Rationale
The research stage recommends submission-required, but submitting the current 0.6652 configuration leaves 0.015 AUC on the table versus the proven lane leader at 0.6801. Per decision rules, we should not collapse to a single low-information calibration tweak when higher-value training/data axes remain open.

The positive prior on the MLP probe head (prefer @ 0.68, lane leader at 0.6801) is the strongest actionable signal. Rather than submitting at 0.6652, we adopt the MLP configuration as the primary submission candidate and open parallel branches for calibration and domain-shift mitigation.

### Branch Portfolio

| Branch | Idea Class | Target | Priority | Rationale |
|--------|-----------|--------|----------|-----------|
| **Primary**: MLP probe head adoption | data_coverage | Reproduce 0.6801 in current pipeline | 60 | Lane leader config; positive prior at 0.68 confidence |
| **Secondary**: Temperature calibration | calibration | Improve ROC-AUC ranking quality | 50 | Prior fusion is miscalibrated (0.662 vs 0.665); temperature scaling on MLP logits is low-risk |
| **Speculative**: Domain-shift augmentation | augmentation | Narrow 0.327 train-val gap | 40 | Frequency-domain augmentation on training soundscapes; high ceiling but uncertain |

### Deprioritized Axes
- **Class coverage expansion in isolation**: Flat metric movement (0.6651→0.6652) proves this axis is exhausted alone. Only revisit post-submission if coverage gap persists.
- **Embedding-level mixup / micro-tuning variants**: Five rounds regressed; three axes exhausted per negative prior. Vetoed.
- **Further MLP architecture variants**: Blocked until submission parity is established per research rejection.

### Submission Stance
**Defer** — Do not submit at 0.6652. Target the MLP probe head at 0.6801 as the submission candidate. If the primary branch reproduces or exceeds 0.6801, proceed to submission immediately. If it regresses, fall back to packaging run-0002-0003 directly.

### Negative Prior Compliance
- Axis exhaustion (avoid @ 0.68): No further micro-tuning on current head. Branch 1 adopts a different proven configuration rather than iterating on the exhausted one. ✅
- No re-running class coverage expansion alone. ✅

### Policy Trace
- `require` (embedding-head complementary signal): All branches use embedding probe heads. ✅
- `prefer` (MLP probe head as baseline): Branch 1 directly adopts this. ✅
- `avoid` (exhausted micro-tuning axes): No branches touch mixup or augmentation variants on the current head. ✅
- `conditional` (coverage gap): Deferred to post-submission phase. ✅
