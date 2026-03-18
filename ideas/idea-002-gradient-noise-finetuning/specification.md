# Specification: Gradient Noise as Regularisation in Low-Data Fine-Tuning (Idea 002)

## 1. Problem Statement

Pre-trained language models (e.g., BERT, RoBERTa) are commonly fine-tuned on downstream tasks
using very few labelled examples (10–500 samples).  In this low-data regime, standard gradient
descent overfits rapidly: training accuracy reaches nearly 100 % while validation accuracy
plateaus or regresses.  Existing mitigation strategies—dropout, weight decay, mixup—help but are
generic and not tailored to the fine-tuning of pre-trained transformers.

This specification covers a research plan to investigate whether deliberately injecting
calibrated Gaussian noise into gradients during fine-tuning acts as an implicit regulariser,
improving generalisation beyond what standard fine-tuning and existing regularisation baselines
can achieve.

---

## 2. Research Question & Hypothesis

**Research Question**  
Does adding calibrated Gaussian noise to parameter gradients during fine-tuning of a pre-trained
transformer in a low-data regime reduce overfitting, measured by the generalisation gap
(train accuracy − test accuracy)?

**Testable Hypothesis**  
*H₀*: Gradient-noised fine-tuning (GN-FT) achieves the same test accuracy and generalisation gap
as standard fine-tuning (STD-FT).  
*H₁*: GN-FT achieves statistically significantly lower generalisation gap and equal or higher
test accuracy compared to STD-FT, in the regime of ≤ 100 labelled examples per class.

**Rejection criterion**: paired t-test p < 0.05 on generalisation gap over ≥ 5 seeds.

---

## 3. Background & Novelty Assessment

### Prior Work
- Neelakantan et al. (2015) – gradient noise for training deep and recurrent networks.
- Jiang et al. (2020) – SMART regularisation for NLP fine-tuning (perturbation in embedding space).
- Aghajanyan et al. (2021) – intrinsic dimensionality explains low-data fine-tuning of LMs.
- Dodge et al. (2020) – fine-tuning instability in low-data regimes.

### Novelty Gap
Neelakantan et al. study gradient noise in *training-from-scratch* settings for LSTM stability.
SMART applies perturbations in *embedding space*, not gradient space.
Neither work systematically studies gradient noise as an explicit anti-overfitting regulariser
specifically in the *fine-tuning* of *transformer* models in *few-shot* classification tasks,
nor do they provide a principled noise schedule linked to the gradient norm.

---

## 4. Proposed Method

### 4.1 Gradient Noise Schedule

At training step $t$, after computing gradient $g_t = \nabla_\theta \mathcal{L}$, we inject:

$$\tilde{g}_t = g_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}\!\left(0,\, \sigma_t^2 I\right)$$

where the noise standard deviation follows an annealed schedule:

$$\sigma_t = \frac{\eta}{\left(1 + t\right)^\gamma}$$

with $\eta$ (initial noise level) and $\gamma$ (decay rate) as hyperparameters.

This schedule ensures large noise early (exploration) that decays to near-zero (exploitation),
analogous to simulated annealing.

### 4.2 Norm-Adaptive Variant

We also consider a norm-adaptive schedule to prevent noise dominating small gradients:

$$\sigma_t = \frac{\eta \cdot \|g_t\|_2}{\left(1 + t\right)^\gamma}$$

---

## 5. Experimental Plan

### 5.1 Datasets & Models
| Dataset | Task | Model |
|---|---|---|
| SST-2 | Sentiment (2-class) | DistilBERT-base |
| AG News | Topic (4-class) | DistilBERT-base |
| TREC | Question type (6-class) | DistilBERT-base |

Training sizes: 10, 50, 100, 500 samples per class (stratified splits).

### 5.2 Baselines
1. STD-FT: standard fine-tuning (no extra regularisation)
2. STD-FT + Dropout (p=0.2)
3. STD-FT + Weight Decay (λ=0.01)
4. SMART (Jiang et al., 2020)
5. **GN-FT (ours)** – fixed noise schedule
6. **GN-FT-Adaptive (ours)** – norm-adaptive noise

### 5.3 Metrics
- Test Accuracy (↑)
- Generalisation Gap = Train Acc − Test Acc (↓)
- F1-Macro (↑)

### 5.4 Hyperparameters
Grid: η ∈ {1e-4, 1e-3, 1e-2}, γ ∈ {0.1, 0.55, 1.0}.

---

## 6. Chronological Flow Diagram

```
START
  │
  ▼
[Step 1] Literature review & novelty check
  │  Pass → proceed; Fail → pivot
  ▼
[Step 2] Hypothesis formulation (pre-registration)
  ▼
[Step 3] Data preparation
  │    3a. Download SST-2, AG News, TREC via HuggingFace datasets
  │    3b. Stratified subsampling to target training sizes
  │    3c. Sanity check: label balance verification
  ▼
[Step 4] Implement GN-FT
  │    4a. Custom optimizer wrapper injecting Gaussian noise
  │    4b. Fixed and adaptive noise schedule variants
  │    4c. Unit tests: noise is injected correctly; σ decays over steps
  ▼
[Step 5] Run experiments
  │    5a. Train all baselines (5 seeds × 4 training sizes × 3 datasets)
  │    5b. Train GN-FT variants with grid search
  │    5c. Log results to JSON
  │  Resource: Kaggle GPU (DistilBERT ~15 min/run on T4)
  ▼
[Step 6] Visualise & analyse
  │    6a. Learning curves (train/val accuracy over epochs)
  │    6b. Generalisation gap vs. training set size
  │    6c. Statistical tests
  ▼
[Step 7] Validate hypothesis
  │  Pass → write paper; Fail → expand η/γ range or try norm-adaptive variant
  ▼
[Step 8] Write paper
  │    Venue: EMNLP / ICLR
  ▼
END
```

---

## 7. Failure & Setback Handling

| Failure Mode | Detection | Recovery |
|---|---|---|
| No generalisation improvement | t-test p ≥ 0.05 | Expand η/γ; try layer-wise noise |
| GN-FT diverges | NaN loss / exploding gradients | Reduce η; add gradient clipping |
| SMART baseline unavailable | Import error | Implement minimal SMART from paper |
| Kaggle GPU quota exceeded | Job timeout | Use CPU with smaller model (DistilBERT already small) |

---

## 8. Resource Plan

| Phase | Compute | Estimated Time |
|---|---|---|
| Baselines (3 datasets × 4 sizes × 5 seeds) | Kaggle T4 | ~4 h |
| GN-FT grid search | Kaggle T4 | ~3 h |
| Analysis + paper | CPU | ~3 h |

---

## 9. Paper Outline

1. Abstract
2. Introduction
3. Background (fine-tuning instability, gradient noise, SMART)
4. Method (gradient noise schedule)
5. Experiments
6. Results
7. Discussion (layer-wise analysis, connection to Langevin dynamics)
8. Related Work
9. Conclusion
10. References
