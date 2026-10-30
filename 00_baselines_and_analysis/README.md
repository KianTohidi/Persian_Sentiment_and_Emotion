# Fine-Tuned Baselines and Additional Analyses

Code and results for the comparison of four zero-shot LLMs (Claude 3.7 Sonnet, DeepSeek-V3, Gemini 2.0 Flash, GPT-4o) with three fine-tuned transformer baselines (ParsBERT, XLM-RoBERTa, multilingual BERT), including the manual label check of texts that all models misclassify.

## Folder structure

```
00_baselines_and_analysis/
├── common.py                shared paths and helper functions
├── 00_...py – 11_...py      scripts, run in numerical order
├── data/
│   ├── input/               cleaned datasets and the balanced test samples
│   ├── splits/              train / validation / test files
│   └── llm_predictions/     predictions of the four LLMs
├── results/                 tables, figure and run log
│   └── predictions/         predictions of the 30 fine-tuning runs
└── label_check/             annotation sheets, consensus labels and summaries
```

## Scripts

| Script | Purpose |
|---|---|
| `00_environment_info.py` | Prints Python, library and GPU versions |
| `01_make_splits.py` | Creates train/validation/test splits (test = balanced evaluation sample) |
| `02_finetune_baselines.py` | Fine-tunes the three baselines (5 seeds, 2 tasks); needs a GPU |
| `03_summarize_runs.py` | Mean and SD over seeds; selects the run with the best validation macro-F1 |
| `04_compare_models.py` | Bootstrap CIs, McNemar tests with Holm correction, seed robustness |
| `05_error_analysis.py` | Per-class F1, frequent confusions, error overlap |
| `06_plot_results.py` | Figure of macro-F1 with confidence intervals |
| `07_label_check_create_sheets.py` | Blind annotation sheets for the label check |
| `08_label_check_agreement.py` | Inter-annotator agreement before discussion |
| `09_consensus_create_sheet.py` | Sheet with the disagreements for consensus labelling |
| `10_consensus_analysis.py` | Consensus labels vs. dataset labels and model predictions |
| `11_appendix_tables.py` | Appendix tables as an Excel file |

Scripts `02`, `07` and `09` create files that are then filled in or used by later steps; the completed files are included, so every later script can be run directly.

### Fine-tuned baselines and additional analyses:
- `00_baselines_and_analysis/`: fine-tuning of ParsBERT, XLM-R and mBERT, comparison with the four LLMs (confidence intervals, McNemar tests, error analysis) and the manual label check of texts misclassified by all models. See the README in that folder.
- `00_annotation_validation/`: annotation validation of the five candidate datasets.
- `00_robustness_check/`: batch (20) vs. single-instance (1) robustness check of the LLM predictions.


## Setup

```
pip install -r requirements.txt
python 01_make_splits.py
```

The reported runs used Google Colab (Python 3.13, Tesla T4 GPU); library versions are in `requirements.txt` and were recorded on 20 October 2026. Only `02_finetune_baselines.py` needs a GPU.

## Notes

- Fine-tuning: 4 epochs, learning rate 2e-5, batch size 16, maximum length 128, class-weighted loss, seeds 42–46; the checkpoint with the best validation macro-F1 is used for testing.
- Invalid LLM answers (labels outside the allowed set) are counted as errors in all comparisons.
- Label check: two annotators labelled the texts independently and blind to the dataset labels; disagreements were resolved by discussion. Annotators are anonymized. The original sheets also had an optional sarcasm column, which was not used.
- The text files are derived from the public datasets listed in the main README; see their licenses.
