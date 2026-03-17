# Log: Gradient Noise as Regularisation in Low-Data Fine-Tuning (Idea 002)

---

## 2026-03-17

### [STEP 1] Literature Review & Novelty Check

**Action**: Searched for "gradient noise fine-tuning", "noise injection BERT low-data",
"few-shot fine-tuning regularisation transformer".

**Findings**:
- Neelakantan et al. (2015): gradient noise improves training stability for deep/recurrent nets
  in *from-scratch* settings; not applied to transformer fine-tuning.
- Jiang et al. (2020) SMART: symmetric KL regularisation in embedding space; different from
  gradient-space noise; targets adversarial robustness more than overfitting.
- Aghajanyan et al. (2021): shows low intrinsic dimensionality of fine-tuning manifold, explaining
  why small interventions matter, but does not propose gradient noise.
- No paper found that proposes a calibrated gradient noise schedule specifically for low-data
  transformer fine-tuning with an overfitting-reduction objective.

**Decision**: Novelty confirmed.  Proceeding to Step 2.

**Novelty Score (internal)**: 6/10 (builds directly on known gradient-noise idea but the
systematic application to transformer fine-tuning in few-shot regime is novel).

---

### [STEP 2] Hypothesis Formulation

**Action**: Defined H₀ and H₁ (see specification.md §2).  Pre-registered acceptance criterion:
paired t-test p < 0.05 on generalisation gap over ≥ 5 seeds, at training set sizes ≤ 100
samples per class.

**Reasoning**: Generalisation gap is the primary metric because the goal is reducing overfitting,
not necessarily boosting absolute accuracy.

---

## TODO (Next Session)

- [ ] Set up HuggingFace datasets and DistilBERT environment.
- [ ] Implement `GradientNoiseOptimizer` wrapper.
- [ ] Run STD-FT baseline on SST-2 (n=50) to establish baseline numbers.
- [ ] Launch grid search on Kaggle GPU.
- [ ] Update this log with results.
