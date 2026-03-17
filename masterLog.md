# Master Research Log

This file is the top-level index for all autonomous research activity in this repository.
It tracks every idea, its current status, key outcomes, and links to idea-specific logs and specifications.

---

## How This Repository Works

```
masterLog.md          ← this file: global index & status board
ideas/
  idea-NNN-<slug>/
    specification.md  ← full plan: flow diagram, step guides, failure handling
    log.md            ← chronological action log with reasoning
    experiments/      ← runnable experiment code
    paper/            ← manuscript (Markdown, LaTeX, PDF)
```

New ideas are added by appending a row to the **Idea Registry** table and creating the
corresponding directory.  The status field progresses through:

| Status | Meaning |
|---|---|
| `PROPOSED` | Idea recorded; novelty not yet evaluated |
| `NOVELTY_OK` | Novelty check passed; hypothesis being developed |
| `HYPOTHESIS` | Testable hypothesis formulated |
| `EXPERIMENTS` | Experiments being implemented/running |
| `RESULTS` | Experiments completed; results being analysed |
| `VALIDATED` | Results validate (or refute) the hypothesis |
| `WRITING` | Research paper being written |
| `SUBMITTED` | Paper submitted to venue |
| `ABANDONED` | Idea abandoned; see log for reason |

---

## Idea Registry

| ID | Slug | One-line Description | Status | Venue Target | Last Updated |
|---|---|---|---|---|---|
| 001 | calibration-aware-kd | Does adding an explicit calibration loss to knowledge distillation improve student ECE without sacrificing accuracy? | `EXPERIMENTS` | NeurIPS / ICML | 2026-03-17 |
| 002 | gradient-noise-finetuning | Does calibrated gradient noise during fine-tuning of pre-trained models reduce overfitting in low-data regimes? | `HYPOTHESIS` | ICLR / EMNLP | 2026-03-17 |

---

## Change Log

| Date | Entry |
|---|---|
| 2026-03-17 | Repository initialised.  Added ideas 001 and 002. Created specification and log files for both ideas. |
