# Log: Calibration-Aware Knowledge Distillation (Idea 001)

All actions, decisions, and reasoning are recorded here in chronological order.
Each entry follows the format: `[YYYY-MM-DD] [STEP N] <Action> — <Reasoning>`.

---

## 2026-03-17

### [STEP 1] Literature Review & Novelty Check

**Action**: Searched Google Scholar and Semantic Scholar for "knowledge distillation calibration",
"calibrated knowledge distillation", "ECE knowledge distillation".

**Findings**:
- Hinton et al. (2015): KD uses soft labels from teacher at temperature τ.  No calibration objective.
- Guo et al. (2017): temperature scaling is effective post-hoc calibration but applied *after*
  training; it does not change the training dynamics.
- Müller et al. (2019): label smoothing improves calibration as a by-product.  The smoothing is
  uniform, not guided by a calibration metric.
- Minderer et al. (2021): modern networks (ViT, MLP-Mixer) are better calibrated than ResNets, but
  KD students are not studied specifically.
- No paper found that adds a differentiable ECE term directly to the KD training loss.

**Decision**: Novelty confirmed.  Proceeding to Step 2.

**Novelty Score (internal)**: 7/10 (incremental but practically important; clear metric for
comparison).

---

### [STEP 2] Hypothesis Formulation

**Action**: Defined H₀ and H₁ (see specification.md §2).  Pre-registered acceptance criterion:
paired t-test p < 0.05, ECE ↓, Top-1 Acc drop < 0.5 pp over ≥ 5 seeds.

**Reasoning**: Pre-registration prevents p-hacking and ensures objective evaluation.

---

### [STEP 3] Experiment Infrastructure

**Action**: Created experiment skeleton:
- `experiments/train_teacher.py` — trains ResNet teacher on CIFAR-10/100.
- `experiments/distill_student.py` — trains student with configurable loss (standard KD / CA-KD).
- `experiments/losses.py` — implements standard KD loss and differentiable CA-KD loss.
- `experiments/evaluate.py` — computes ECE-15, MCE, NLL, Brier Score, Top-1 Acc.
- `experiments/visualise.py` — reliability diagrams and Pareto plots.
- `experiments/analyse.py` — loads result JSONs, runs t-tests, prints summary table.
- `experiments/requirements.txt` — Python dependencies.
- `experiments/tests/test_losses.py` — unit tests for loss functions.

**Reasoning**: Clean separation of concerns makes it easy to swap components (e.g., try a different
calibration loss) without rewriting the training loop.

**Status**: Code scaffolded.  Next: run on CIFAR-10 to verify sanity.

---

## TODO (Next Session)

- [ ] Run `train_teacher.py` on CIFAR-10; verify ResNet-56 accuracy ≥ 93.0 %.
- [ ] Run `distill_student.py --method standard_kd`; verify ResNet-20 accuracy ≥ 91.0 %.
- [ ] Run unit tests for `losses.py`.
- [ ] Launch λ grid search on Kaggle GPU.
- [ ] Update this log with results.
