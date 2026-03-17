# Specification: Calibration-Aware Knowledge Distillation (Idea 001)

## 1. Problem Statement

Knowledge distillation (KD) transfers the "dark knowledge" of a large teacher network into a smaller
student network by minimising the KL divergence between teacher and student soft-label distributions.
A well-documented side-effect is that the student can *inherit the teacher's miscalibration*: if the
teacher is overconfident, the student becomes overconfident too.  This specification describes a
research plan to

1. empirically confirm that standard KD propagates teacher miscalibration,
2. propose a calibration-aware distillation loss that simultaneously minimises task loss and reduces
   Expected Calibration Error (ECE),
3. demonstrate statistically significant improvements in ECE across multiple benchmark/architecture
   pairs without meaningful accuracy degradation.

---

## 2. Research Question & Hypothesis

**Research Question**  
Does adding an explicit calibration regularisation term to the standard knowledge-distillation
objective improve student model calibration (ECE ↓) without significantly degrading classification
accuracy (Top-1 Acc)?

**Testable Hypothesis**  
*H₀*: A student trained with the calibration-aware KD loss (CA-KD) achieves the same ECE as a
student trained with standard KD.  
*H₁*: A student trained with CA-KD achieves a statistically significantly lower ECE than a student
trained with standard KD, with no more than 0.5 pp drop in Top-1 accuracy.

**Null hypothesis rejection criterion**: paired t-test p < 0.05 over ≥ 5 independent training runs.

---

## 3. Background & Novelty Assessment

### Prior Work
- Hinton et al. (2015) – original KD formulation.
- Guo et al. (2017) – temperature scaling as post-hoc calibration (ICML 2017).
- Müller et al. (2019) – label smoothing improves calibration (NeurIPS 2019).
- Moon et al. (2020) – Inter-Class Relationship (IKD) for knowledge distillation.
- Minderer et al. (2021) – revisiting calibration of modern neural networks (NeurIPS 2021).

### Novelty Gap
No prior work jointly optimises *task accuracy* and *calibration quality* **during** the
distillation training loop itself using an explicit, theoretically grounded calibration loss term.
Temperature scaling is purely post-hoc.  Label smoothing targets calibration implicitly.
CA-KD is the first distillation objective that makes calibration a first-class training objective.

---

## 4. Proposed Method

### 4.1 Standard KD Loss
```
L_KD = α · L_CE(y, p_s) + (1−α) · τ² · KL(p_t^τ ‖ p_s^τ)
```
where τ is the temperature, p_s/p_t are student/teacher softmax distributions, y is the one-hot label.

### 4.2 Calibration Loss Term
We add a differentiable approximation of the Expected Calibration Error:
```
L_cal = Σ_b (|B_b| / n) · |acc(B_b) − conf(B_b)|
```
where B_b are confidence bins (soft-binning via a sigmoid ramp to maintain differentiability).

### 4.3 CA-KD Objective
```
L_CAKD = L_KD + λ · L_cal
```
λ is a hyperparameter controlling the calibration-accuracy trade-off.

---

## 5. Experimental Plan

### 5.1 Datasets & Architectures
| Dataset | Teacher | Student |
|---|---|---|
| CIFAR-10 | ResNet-56 | ResNet-20 |
| CIFAR-100 | ResNet-110 | ResNet-32 |
| Tiny-ImageNet | ResNet-50 | MobileNetV2 |

### 5.2 Baselines
1. Cross-Entropy only (no distillation)
2. Standard KD (Hinton et al.)
3. KD + Temperature Scaling (post-hoc)
4. KD + Label Smoothing (LS=0.1)
5. **CA-KD (ours)**

### 5.3 Metrics
- Top-1 Accuracy (↑)
- Expected Calibration Error – ECE-15 (↓, 15 equal-width bins)
- Maximum Calibration Error – MCE (↓)
- Negative Log-Likelihood – NLL (↓)
- Brier Score (↓)

### 5.4 Hyperparameter Search
Grid search over λ ∈ {0.01, 0.05, 0.1, 0.5, 1.0} using 20 % of training data as validation.
Best λ selected per dataset/architecture pair.

### 5.5 Statistical Significance
5 independent runs per configuration (different random seeds).
Paired t-test comparing standard KD vs. CA-KD on ECE and Top-1.

---

## 6. Chronological Flow Diagram

```
START
  │
  ▼
[Step 1] Literature review & novelty check
  │  Pass → novelty confirmed
  │  Fail → abandon / pivot idea
  ▼
[Step 2] Formulate testable hypothesis (H₀ / H₁)
  │
  ▼
[Step 3] Implement baseline models
  │  Subtasks:
  │    3a. Data loaders (CIFAR-10/100, Tiny-ImageNet)
  │    3b. Teacher architectures + pretrained weights
  │    3c. Standard KD training loop
  │  Sanity check: teacher accuracy ≥ published numbers ± 0.3 pp
  │  Fail → debug data pipeline / hyperparameters
  ▼
[Step 4] Implement CA-KD loss
  │    4a. Differentiable soft-bin ECE
  │    4b. Combined CA-KD objective
  │    4c. Unit-test: L_cal = 0 for a perfectly calibrated model
  │  Fail → fix differentiable ECE implementation
  ▼
[Step 5] Run experiments
  │    5a. Train all baselines (5 seeds each)
  │    5b. Grid search λ
  │    5c. Train CA-KD with best λ (5 seeds)
  │  Resource: Kaggle GPU (~1-2 h per run)
  │  Fail/timeout → reduce epochs / use smaller dataset first
  ▼
[Step 6] Analyse & visualise results
  │    6a. Reliability diagrams (calibration curves)
  │    6b. Accuracy–ECE Pareto plot
  │    6c. Statistical tests
  │  Fail (no significant improvement) → see §8
  ▼
[Step 7] Validate hypothesis
  │    Pass: p < 0.05, ECE ↓, Acc drop < 0.5 pp
  │    Fail: revisit λ range or loss formulation → loop to Step 4/5
  ▼
[Step 8] Write research paper
  │    8a. Draft in Markdown
  │    8b. Convert to LaTeX (conference template)
  │    8c. Generate plots, include in paper
  │    8d. Proofread & validate references
  │    8e. Build PDF
  ▼
END (SUBMITTED)
```

---

## 7. Step-by-Step Navigation Guide

### Step 1 – Literature Review
- Query: Google Scholar / Semantic Scholar for "knowledge distillation calibration".
- Record all relevant papers in `experiments/references.bib`.
- Decision: if a paper already proposes an identical method, pivot to a complementary angle
  (e.g., different modality, different loss form).

### Step 2 – Hypothesis Formulation
- Write H₀ and H₁ in `log.md`.
- Define the acceptance/rejection criterion before running any experiment (pre-registration).

### Step 3 – Baselines
- Run `python experiments/train_teacher.py --dataset cifar10` first.
- Verify accuracy against published numbers.
- Run `python experiments/distill_student.py --method standard_kd`.

### Step 4 – CA-KD Implementation
- See `experiments/losses.py` for the differentiable calibration loss.
- Unit test: `python -m pytest experiments/tests/test_losses.py -v`.

### Step 5 – Experiments
- Use Kaggle notebook or remote server.
- Log each run to `experiments/results/run_<seed>_<method>.json`.
- On resource exhaustion: reduce batch size, use mixed precision.

### Step 6 – Visualisation
- `python experiments/visualise.py --results_dir experiments/results/`
- Outputs: `experiments/figures/*.png`

### Step 7 – Validation
- `python experiments/analyse.py` prints t-test results and summary table.
- If H₀ is not rejected: increase λ search range, try focal calibration loss variant.

### Step 8 – Paper Writing
- Template: NeurIPS 2025 (or ICML).  Files in `paper/`.
- `paper/main.tex` is the entry point.
- Build: `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

---

## 8. Failure & Setback Handling

| Failure Mode | Detection | Recovery Action |
|---|---|---|
| Teacher accuracy below published baseline | Sanity check in Step 3 | Debug data augmentation, learning rate schedule; re-run |
| CA-KD loss NaN / divergence | Training loss monitoring | Clip gradients; reduce λ; use log-space ECE |
| No ECE improvement (p ≥ 0.05) | Step 7 statistical test | Expand λ grid; try soft-binning with more bins; test on more architectures |
| Accuracy drop > 0.5 pp | Metric comparison table | Reduce λ; anneal λ during training |
| Experiment timeout on Kaggle | Wall-clock monitor | Reduce epochs; use CIFAR-10 only; use gradient checkpointing |
| Related work overlap found post-hoc | Literature re-check | Pivot to novel angle (different modality, loss, regime) or position as complementary |
| LaTeX build failure | CI / local build | Check bibliography, missing packages; use `latexmk -pdf` for dependency resolution |

---

## 9. Resource Plan

| Phase | Compute | Estimated Time |
|---|---|---|
| Baselines (CIFAR-10, 200 epochs × 5 seeds) | Kaggle T4 GPU | ~3 h |
| CA-KD grid search (CIFAR-10, 5 λ × 5 seeds) | Kaggle T4 GPU | ~6 h |
| Full evaluation (all datasets/architectures) | Kaggle T4 GPU | ~12 h |
| Paper writing | CPU | ~4 h |

---

## 10. Paper Outline

1. Abstract
2. Introduction (motivation, contributions)
3. Background (KD, calibration, ECE)
4. Method (CA-KD loss derivation)
5. Experiments (datasets, baselines, metrics)
6. Results (tables, reliability diagrams)
7. Discussion (ablation: λ sensitivity, soft-bin width)
8. Related Work
9. Conclusion
10. References
11. Appendix (additional results, proofs)
