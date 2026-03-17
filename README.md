# autoresearch

An autonomous AI/ML research repository.  It continuously finds novel research ideas, implements
experiments, validates results, and writes conference-ready papers.

---

## Repository Structure

```
masterLog.md          ← global index of all research ideas and their status
ideas/
  idea-NNN-<slug>/
    specification.md  ← full research plan: flow diagram, step guide, failure handling
    log.md            ← chronological action log with reasoning
    experiments/      ← runnable Python experiment code
      requirements.txt
      *.py            ← data loaders, models, losses, training loops, evaluation
      tests/          ← pytest unit tests
      results/        ← JSON experiment results (gitignored by default)
      figures/        ← generated plots (gitignored by default)
    paper/            ← manuscript sources (Markdown + LaTeX + references)
      paper.md        ← draft in Markdown
      main.tex        ← LaTeX source (conference template)
      references.bib  ← BibTeX bibliography
```

---

## Current Research Ideas

| ID | Short Name | Status |
|---|---|---|
| 001 | [Calibration-Aware Knowledge Distillation](ideas/idea-001-calibration-aware-kd/) | `EXPERIMENTS` |
| 002 | [Gradient Noise Fine-Tuning](ideas/idea-002-gradient-noise-finetuning/) | `HYPOTHESIS` |

See [`masterLog.md`](masterLog.md) for the full status board.

---

## Running an Experiment

```bash
# Install dependencies (inside the idea's experiments/ directory)
pip install -r ideas/idea-001-calibration-aware-kd/experiments/requirements.txt

# Run unit tests
cd ideas/idea-001-calibration-aware-kd/experiments
python -m pytest tests/ -v

# Train teacher
python train_teacher.py --dataset cifar10 --arch resnet56 --epochs 200 --seed 42

# Distil student (standard KD)
python distill_student.py --dataset cifar10 --teacher_arch resnet56 --student_arch resnet20 \
    --method standard_kd --seed 42

# Distil student (CA-KD)
python distill_student.py --dataset cifar10 --teacher_arch resnet56 --student_arch resnet20 \
    --method cakd --lambda_cal 0.1 --seed 42

# Evaluate
python evaluate.py --dataset cifar10 --arch resnet20 \
    --checkpoint results/student_cifar10_resnet20_standard_kd_seed42.pth \
    --output results/eval_student_cifar10_resnet20_standard_kd_seed42.json

# Visualise
python visualise.py --results_dir results/ --figures_dir figures/

# Statistical analysis
python analyse.py --results_dir results/
```

---

## Process Overview

1. **Idea Discovery** – novel, tractable AI/ML problems are identified and logged in `masterLog.md`.
2. **Novelty Evaluation** – prior-work search; confirmed in `log.md`.
3. **Hypothesis Formulation** – testable H₀/H₁ with pre-registered acceptance criteria.
4. **Experiment Implementation** – self-contained Python code with unit tests.
5. **Experiment Execution** – runs on Kaggle GPU / remote CPU server; results in JSON.
6. **Visualisation** – reliability diagrams, Pareto plots, learning curves.
7. **Validation** – statistical significance tests; hypothesis confirmed or refuted.
8. **Paper Writing** – Markdown draft → LaTeX (conference template) → PDF.
