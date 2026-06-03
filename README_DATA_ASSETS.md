# AgriBot-BD Dataset and Scripts Asset Bundle

This directory (`data_assets/`) contains the complete set of scripts, notebooks, and dataset splits used for synthesizing, fine-tuning, and evaluating the AgriBot-BD instruction-tuning pipeline.

This repository structure is designed for immediate open-source publication on GitHub and HuggingFace.

## Directory Structure

### 1. `scripts/`
This folder contains the consolidated end-to-end Python scripts used to build and evaluate the project. They should be run sequentially:

- **`01_glossary_generation.py`**: Extracts the foundational agricultural terminology from authoritative sources.
- **`02_qa_generation.py`**: Synthesizes the core QA pairs using the Partitioned Seed Generation Matrix (PSGM) technique. This produces the dual-structured, citation-grounded outputs across varied query registers.
- **`03_dataset_formatting.py`**: Cleans the raw synthesized JSONL files, enforces the translation glossary, removes empty responses, formats them into OpenAI's ChatML structure, and splits the data into Train/Val/Test subsets.
- **`04_evaluation_metrics.py`**: The definitive evaluation script. It assesses the predictions using ROUGE-L, BERTScore F1, Citation Formatting Compliance, and the Practical Hallucination Rate.
- **`convert_to_parquet.py`**: Utility to convert JSONL/JSON files to compressed Parquet format for HuggingFace upload.

### 2. `EVALUATION_METHODOLOGY.md`
A comprehensive methodology document explaining the rationale, mathematical formulation, and consolidated results of our normalized Practical Semantic Hallucination evaluation framework across all 5 models.

### 3. `dataset_hf/`
This folder contains the finalized dataset splits formatted in pure ChatML, ready for direct upload to a HuggingFace Dataset repository. Each file is available in both JSONL and Parquet formats.

| File | Format | Records | Description |
|------|--------|---------|-------------|
| `train.jsonl/.parquet` | ChatML | 128,336 | QLoRA Fine-tuning split |
| `val.jsonl/.parquet` | ChatML | 4,621 | Validation split |
| `test.jsonl/.parquet` | ChatML | 4,620 | Held-out test set |
| `full_dataset.jsonl/.parquet` | Raw | 145,817 | Complete unfiltered dataset |
| `instructions_unified.jsonl/.parquet` | Unified | 17,866 | Instruction-formatted data |
| `safety_qas.jsonl/.parquet` | ChatML | 1,000 | Safety/adversarial refusal QAs |
| `farmer_benchmark.json/.parquet` | JSON | 981 | Real-world farmer evaluation queries |
| `knowledge_nodes.json/.parquet` | JSON | 290 | Knowledge Node metadata |

### 4. `notebooks/`
- **`agriml_gemma4_finetuning.ipynb`**: The primary Jupyter notebook demonstrating the parameter-efficient fine-tuning (PEFT/QLoRA) of the Gemma 4 E2B model using the `train.jsonl` dataset.

### 5. `chemicals/`
Chemical whitelist data (276 registered active ingredients, 707 aliases) compiled from Bangladesh DAE guidelines.

### 6. `glossary/`
The 1,417-term agricultural glossary mapping English technical terms to formally accepted Bengali equivalents.

## License
This dataset is released under **CC-BY-4.0** (see `LICENSE`).

## How to Use
To replicate the exact pipeline described in the paper:
1. Ensure your `GEMINI_API_KEY` is configured in your environment.
2. Run the scripts in the `scripts/` folder in numerical order.
3. Once the datasets are generated and formatted in `dataset_hf/`, upload them to HuggingFace or run the `notebooks/agriml_gemma4_finetuning.ipynb` directly to reproduce the fine-tuned model weights.
4. Run the inference pipeline and evaluate the predictions using `scripts/04_evaluation_metrics.py`.
