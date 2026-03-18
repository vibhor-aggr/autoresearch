# autoresearch

An autonomous AI/ML research repository.  The **pipeline** continuously finds novel research ideas,
implements experiments, validates results, and writes conference-ready papers — all driven by an LLM.

---

## Automated Pipeline

```
idea generation
      │
      ▼
hypothesis generation
      │
      ▼
experiment planning  (LLM generates runnable Python code)
      │
      ▼
experiment execution  (local | Kaggle | remote SSH)
      │
      ▼
validation & evaluation  (statistical tests)
      │
   pass?
  ┌────┴────┐
  │         │
  ▼         ▼ (revise)
paper   ┌── back to idea / hypothesis / experiment planning
writing └──────────────────────────────────────────────────
```

### Quick start

```bash
# 1. Install pipeline dependencies
pip install -r pipeline/requirements.txt

# 2. (Optional) configure LLM + backend
cp config/pipeline.yaml config/pipeline_local.yaml
# edit config/pipeline_local.yaml  – set provider, model, api_key_env

# 3. Run one idea end-to-end (stub LLM, local backend – no API key needed)
python run_pipeline.py

# 4. Run with a real LLM (OpenAI)
OPENAI_API_KEY=sk-... python run_pipeline.py \
    --llm-provider openai --llm-model gpt-4o

# 5. Use Kaggle for experiment execution
python run_pipeline.py --backend kaggle

# 6. Use a remote server
python run_pipeline.py --backend remote

# 7. Run the pipeline test suite
python -m pytest pipeline/tests/ -v
```

### Command-line options

```
python run_pipeline.py [OPTIONS]

  --config PATH            Path to YAML config (default: config/pipeline.yaml)
  --llm-provider PROVIDER  openai | anthropic | stub
  --llm-model MODEL        e.g. gpt-4o, claude-3-5-sonnet-20241022
  --backend BACKEND        local | kaggle | remote
  --max-ideas N            Stop after N ideas (0 = unlimited)
  --ideas-dir PATH         Override the ideas/ directory
  --log-level LEVEL        DEBUG | INFO | WARNING | ERROR
```

---

## Repository Structure

```
run_pipeline.py       ← CLI entry point
config/
  pipeline.yaml       ← default pipeline config (LLM, backend, validation thresholds)
  backends.yaml.example ← credential template (never commit the real file)
pipeline/
  orchestrator.py     ← PipelineOrchestrator – main loop
  config.py           ← PipelineConfig dataclasses + YAML loader
  state.py            ← IdeaState, PipelineStage, RevisionTarget, JSON persistence
  stages/
    idea_generation.py  ← Stage 1: generate novel ideas via LLM
    hypothesis.py       ← Stage 2: formulate testable H₀/H₁
    experiment_plan.py  ← Stage 3: plan + generate experiment code
    experiment_run.py   ← Stage 4: execute on chosen backend
    validation.py       ← Stage 5: statistical validation, pass/fail, revision target
    paper_writing.py    ← Stage 6: generate Markdown + LaTeX paper
  backends/
    local.py            ← LocalBackend  (subprocess on current machine)
    kaggle_backend.py   ← KaggleBackend (Kaggle Notebooks API)
    remote_backend.py   ← RemoteSSHBackend (Paramiko SSH)
  utils/
    llm.py              ← LLMClient (openai | anthropic | stub)
    file_ops.py         ← Markdown helpers, directory scaffolding
  tests/
    test_state.py       ← state machine unit tests
    test_orchestrator.py ← end-to-end pipeline tests (stub LLM)
    test_backends.py    ← backend + utility tests
masterLog.md          ← global index of all research ideas and their status
ideas/
  idea-NNN-<slug>/
    specification.md  ← full research plan: flow diagram, step guide, failure handling
    log.md            ← chronological action log with reasoning
    pipeline_state.json ← current pipeline state (gitignored)
    experiments/      ← runnable Python experiment code
      run_experiment.py ← generated experiment script
      requirements.txt
      tests/          ← pytest unit tests
      results/        ← JSON experiment results (gitignored)
      figures/        ← generated plots (gitignored)
    paper/            ← manuscript sources
      paper.md        ← Markdown draft
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

## Compute Backends

| Backend | When to use | Prerequisites |
|---|---|---|
| `local` | Agent mode, GitHub Codespaces, any local machine | none |
| `kaggle` | GPU experiments on Kaggle Notebooks | `pip install kaggle` + `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars |
| `remote` | Your own GPU server | `pip install paramiko` + SSH key access |

---

## Configuration

Edit `config/pipeline.yaml`:

```yaml
llm:
  provider: openai          # openai | anthropic | stub
  model: gpt-4o
  api_key_env: OPENAI_API_KEY

backend:
  type: local               # local | kaggle | remote

validation:
  significance_threshold: 0.05
  max_accuracy_drop_pp: 0.5
  max_revision_rounds: 3

pipeline:
  continuous: true
  max_ideas: 10
```

---

## Manually Running an Existing Experiment

```bash
# Install dependencies (inside the idea's experiments/ directory)
pip install -r ideas/idea-001-calibration-aware-kd/experiments/requirements.txt

# Run unit tests
cd ideas/idea-001-calibration-aware-kd/experiments
python -m pytest tests/ -v

# Train teacher
python train_teacher.py --dataset cifar10 --arch resnet56 --epochs 200 --seed 42

# Evaluate
python evaluate.py --dataset cifar10 --arch resnet20 \
    --checkpoint results/student_cifar10_resnet20_standard_kd_seed42.pth \
    --output results/eval_student_cifar10_resnet20_standard_kd_seed42.json
```

---

## Process Overview

1. **Idea Generation** – LLM proposes novel, tractable AI/ML problems; novelty assessed vs existing ideas.
2. **Hypothesis Formulation** – testable H₀/H₁ with pre-registered acceptance criteria.
3. **Experiment Planning** – LLM generates a self-contained `run_experiment.py` and `requirements.txt`.
4. **Experiment Execution** – script runs on the chosen backend; results saved as JSON.
5. **Validation** – paired t-test + LLM interpretation; decides pass / revise / abandon.
6. **Paper Writing** – if validated, LLM writes a full Markdown + LaTeX conference paper.
7. **Revision Loop** – on failure, the orchestrator loops back to idea, hypothesis, or experiment
   planning (up to `max_revision_rounds` times before abandoning).
