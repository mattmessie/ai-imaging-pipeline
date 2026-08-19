# Hybrid Biomedical Image-Analysis Pipeline

A local, hybrid biomedical image-analysis system for the assigned modality:
**fluorescence microscopy (DAPI-style stained nuclei)** — a small synthetic
dataset (112 images) in the spirit of the 2018 Data Science Bowl nuclei set,
with exact ground-truth binary and instance masks.

The pipeline combines a local multimodal LLM (`llama3.2-vision` via
Ollama), classical image processing (Otsu thresholding + regionprops), a
trained U-Net (PyTorch), and local text LLMs (`llama3.2` via Ollama) into
one auditable per-image record: `raw image → segmentation → quantitative
region features → structured JSON record → short narrative`.

**Full report:** `reports/report.pdf` (max 4 pages, per the assignment
brief — see there for the full write-up, the 5 required questions, and
critical discussion of all results below).

## Headline results

| | |
|---|---|
| U-Net validation Dice / IoU | **0.9828 / 0.9662** (10 epochs) |
| U-Net vs. Otsu, pixel-level Dice (test set) | U-Net wins 10/12 images (mean 0.9815 vs 0.9784) |
| U-Net vs. Otsu, object count (test set) | Tied — see note below, this isn't a fair comparison |
| Task 1 naive vs. optimised VLM prompt | Naive drifted into unverifiable biological claims; optimised stayed observational |
| Task 4 hybrid pipeline (12 unseen test images) | `density_class` was uniformly "moderate" on every image despite `n_objects` ranging 8–42 — see report Q4 |

**Important, non-obvious finding:** object counts (via connected-component
labeling) are nearly identical between Otsu and the U-Net. This is *not*
because the U-Net failed — it's because the ground-truth **binary** masks
themselves merge touching/overlapping nuclei into single connected
components (only the separate 16-bit instance-label masks distinguish
them), and the U-Net was trained on binary masks. Both methods are
therefore bound by the same structural limitation for object counting.
Pixel-level Dice/IoU (the metric each model actually optimises) is the
fair comparison, and there the U-Net does show a consistent, if modest,
improvement. This matches the course lecture's own "Common Failure Modes"
guidance: fixing instance separation needs watershed post-processing or an
instance-segmentation head, not a better binary segmenter.

## Repository structure

```text
ai-imaging-pipeline/
│
├── data/nuclei_dataset/          # train (80) / val (20) / test (12) / test_corrupted (4)
│                                  # images, masks (binary), labels (16-bit instance), metadata.csv
│
├── src/imaging_pipeline/
│   ├── config.py                  # paths, constants (image size, model names)
│   ├── data_prep.py                # grayscale conversion, resize, EDA
│   ├── vlm_description.py          # Task 1: multimodal LLM prompting (naive vs optimised)
│   ├── classical_features.py       # Task 2: Otsu, regionprops, deterministic summaries
│   ├── llm_utils.py                # shared: text-only Ollama calls, JSON+narrative parsing
│   ├── llm_interpretation.py       # Task 2: numbers-first LLM prompt
│   ├── unet.py                     # Task 3: U-Net architecture (matches course lab skeleton)
│   ├── train_unet.py               # Task 3: dataset loading, losses, metrics, training loop
│   └── hybrid_pipeline.py          # Task 4: full pipeline (U-Net -> features -> LLM JSON+narrative)
│
├── scripts/                       # one runnable script per pipeline stage (see below)
├── colab/Task1_VLM_Colab.ipynb    # Colab fallback for the vision-model step -- see IMPORTANT note below
├── outputs/{figures,metrics,records,model_objects}/   # every output referenced in the report
├── reports/report.pdf             # the submitted report
└── tests/                         # pytest -- 54 tests, run from a fresh clone to confirm
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Requires a local [Ollama](https://ollama.com) install with `llama3.2-vision`
and `llama3.2` pulled:
```bash
ollama pull llama3.2-vision
ollama pull llama3.2
```

##  Important: known Ollama bug affecting `llama3.2-vision`

**As of Ollama v0.30.0 (May 2026), the `mllama` architecture that
`llama3.2-vision` needs was dropped from Ollama's new inference engine and
has not been restored.** This is a real, currently open upstream bug
(confirmed via multiple GitHub issues, e.g. `ollama/ollama#16547`,
`#16490`), not something specific to any one machine — it was reproduced
identically on a fresh macOS install and a fresh Google Colab install
during development of this project, both giving:
```
error loading model: unknown model architecture: 'mllama'
```

**This was independently confirmed by the module tutor.** An email from
Nickolay Korabel to the cohort states: *"If you encounter any issues when
using the llama3.2-vision model for assignment 3 (e.g. an 'unknown model
architecture: mllama' error), you can use an alternative model instead
(e.g. Qwen2.5-VL/Qwen3-VL or ministral-3:14b vision model), or use the
llama3.2-vision model in Colab (see Lab 2 notebook for details)."* This
matches the diagnosis and the Colab workaround below exactly, and was
reached independently before that email was received.

**If `ollama --version` reports 0.30.0 or later and Task 1's VLM step
fails with this error, that's why — not a bug in this code.** Fixes, in
order of preference for this project:

1. **Use `colab/Task1_VLM_Colab.ipynb`** (the approach actually used to
   produce this project's real Task 1 results), which installs its own
   pinned Ollama server inside a disposable Colab runtime and doesn't
   touch your local install at all. Upload it to
   [colab.research.google.com](https://colab.research.google.com),
   Runtime → Run all, upload the representative image when prompted, and
   the required output files download automatically at the end.
2. **Or downgrade Ollama** to a pre-0.30.0 release (e.g. `v0.22.1`, a
   confirmed stable release from before the regression):
   ```bash
   curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.22.1 sh
   ```
3. **Or swap in an alternative vision model** the tutor's email names as
   acceptable (Qwen2.5-VL/Qwen3-VL, ministral-3:14b) by changing
   `VLM_MODEL` in `src/imaging_pipeline/config.py` and re-pulling that
   model via Ollama — not used for this project's actual submitted
   results (Colab with the real `llama3.2-vision` was used instead), but
   a valid fallback per the tutor's guidance if needed.

This only affects `llama3.2-vision` (the Task 1 multimodal step). The
text-only model `llama3.2` (Tasks 2 and 4) is unaffected and runs
normally on any Ollama version.

## Running the pipeline

Run in order (each stage's outputs feed the next):

```bash
# Task 1: data prep + EDA (no Ollama needed)
python scripts/run_task1_eda.py

# Task 1: multimodal LLM prompting (needs llama3.2-vision -- see note above)
python scripts/run_task1_vlm.py
# -- or, if the mllama bug applies to you: run colab/Task1_VLM_Colab.ipynb instead

# Task 2: classical features (no Ollama needed)
python scripts/run_task2_classical.py
# Task 2: numbers-first LLM interpretation (needs llama3.2, unaffected by the bug above)
python scripts/run_task2_llm.py

# Task 3: train and evaluate the U-Net (no Ollama needed)
python scripts/run_task3_unet.py
# Task 3 (extra): direct U-Net vs Otsu comparison on the test set, for report Q2
python scripts/run_task3_vs_otsu_comparison.py

# Task 4: hybrid pipeline on the 12 unseen test images
python scripts/run_task4_stage1.py       # U-Net + features + prompts (no Ollama needed)
python scripts/run_task4_llm_batch.py    # the actual LLM calls (needs llama3.2)
```

## Running the tests

```bash
pytest
```

54 tests across data preprocessing, Otsu segmentation, the U-Net
architecture and training loop (including Dice/IoU sanity checks on
synthetic perfect/zero-overlap cases), and every LLM-facing module's
prompt construction and JSON/narrative parsing logic — the parsing and
prompt-building tests run without Ollama installed at all (the `ollama`
import is deferred to call-time specifically so this works), so a
reviewer without Ollama set up can still verify the logic is correct.

## Notes on design choices

- **Grayscale throughout.** Task 1 converts to grayscale; this same
  single-channel representation feeds Otsu (Task 2), the U-Net (Task 3,
  `in_ch=1`), and feature extraction (Task 4) — one consistent
  intensity representation across the whole pipeline, not a different
  preprocessing path per task.
- **Numbers-only LLM prompts (Tasks 2 and 4).** The LLM never sees the
  image for the "numbers-first" steps — only the deterministic text
  summary generated from regionprops. This is deliberate, matching the
  brief's explicit instruction, and is exactly what makes it possible to
  compare "what does the model conclude from numbers alone" against "what
  does it conclude from the image directly" (Task 1) as two genuinely
  different information conditions.
- **U-Net architecture is unmodified from the course lab skeleton**
  (`LAB_CNN_unet_segmentation.ipynb`) — same `DoubleConv` blocks, same
  3-level encoder + bottleneck, same `base=16` channel count, same
  combined BCE+Dice loss, same 10-epoch/Adam/lr=1e-3 training recipe. Only
  the `Dataset` class changed, to load the real nuclei images instead of
  the lab's synthetic on-the-fly generator.
- **Checkpointed training.** `train_unet()` supports resuming from a
  saved checkpoint (`outputs/model_objects/unet_checkpoint.pth`), useful
  since a full 10-epoch run can take a few minutes on CPU.
