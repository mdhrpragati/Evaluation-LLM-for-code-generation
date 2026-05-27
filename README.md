
# LLM-as-a-Judge for Code Generation and Benchmark Evaluation

This repository implements a dual-pipeline evaluation framework using the **LLM-as-a-Judge paradigm** to assess both:

- Code generation model performance  
- Benchmark dataset quality  

The goal is to go beyond traditional execution-based metrics (e.g., pass@k) and enable deeper evaluation using reasoning-based judgments from LLMs.

---

## Overview

Large Language Models (LLMs) have significantly improved code generation capabilities. However, evaluating their outputs using only execution-based metrics fails to capture:

- reasoning correctness  
- robustness  
- clarity  
- partial correctness  

This project introduces:

- LLM-based evaluation of generated code  
- LLM-based auditing of benchmark datasets  
- Multi-dimensional evaluation  

---

## Repository Structure

- LLmasajudge_benchmarkevaluationmatchingdataset.py — Benchmark evaluation pipeline  
- llmmodel_evaluationwithjudgemodel.py — Model output evaluation pipeline  
- plotresults.py — Plots for model evaluation  
- generateresults_benchmarkquality.py — Plots for benchmark evaluation  
- evaluation_metrics_qwenjudge.xlsx — Aggregated evaluation metrics  
- evaluation_explanations_qwenjudge.xlsx — Detailed LLM explanations  
- audit_resultsfirst_Qwen_*.xlsx — Benchmark audit results (Qwen)  
- audit_resultsfirst_mistral_*.xlsx — Benchmark audit results (Mistral)  
- audit_resultsfirst_deepseek_*.xlsx — Benchmark audit results (DeepSeek)  
- README.md — Project documentation  

---

## Requirements

- Python 3.9+
- PyTorch  
- Transformers (HuggingFace)  
- Datasets (HuggingFace)  
- Pandas  
- NumPy  
- tqdm  

Install dependencies:

```bash
pip install torch transformers datasets pandas numpy tqdm
```

## How to Run

This section explains how to run both evaluation pipelines and generate results.

---

### 1. Model Output Evaluation

Run the following script:

```bash
python llmmodel_evaluationwithjudgemodel.py
```

This will:

- Load code generation models  
- Generate **5 candidate solutions per problem**  
- Extract and clean valid Python functions  
- Evaluate each solution using an LLM judge  
- Compute performance metrics such as:
  - average score  
  - maximum score  
  - diversity  

Generated output files:

- `evaluation_metrics_qwenjudge.xlsx` — aggregated performance metrics  
- `evaluation_explanations_qwenjudge.xlsx` — detailed explanations for each evaluated solution  

---

### 2. Benchmark Quality Evaluation

Run the following script:

```bash
python LLmasajudge_benchmarkevaluationmatchingdataset.py
```

This will:
- Load HumanEval and HumanEvalNext datasets
- Match tasks across datasets using:
  - task identifiers, or
  - function name matching
- Ensure evaluation is performed on identical problem instances
- Use an LLM judge to evaluate each task
- Assign scores for:
  - correctness of test cases
  - coverage of edge cases
  - solution quality
  - clarity of problem description

Generated output files:

- audit_resultsfirst_Qwen_*.xlsx
- audit_resultsfirst_mistral_*.xlsx
- audit_resultsfirst_deepseek_*.xlsx

### 3. Generate Plots:

Run the following script:

```bash
python plotresults.py
```

This will generate:

- Model comparison graphs
- Average score visualizations
- Performance trends across datasets

### 4. Benchmark Evaluation Results

Run the following script:

```bash
python generateresults_benchmarkquality.py
```

This will generate:

- HumanEval vs HumanEvalNext comparison plots
- Benchmark quality evaluation graphs across dimensions

---

## Overall Process

This project follows a dual evaluation pipeline:

### Model Output Evaluation
- Generates multiple candidate solutions per problem  
- Uses an LLM to evaluate correctness  
- Computes aggregate metrics such as average score and diversity  

### Benchmark Quality Evaluation
- Matches tasks across datasets  
- Evaluates dataset quality using LLM reasoning  
- Scores datasets across multiple dimensions  

---

## Datasets Used

- HumanEval  
- HumanEvalNext  

---

## Notes

- GPU is recommended for faster execution  
- Models and datasets will be downloaded automatically  
- HuggingFace caching is used internally  

---

## Reproducibility

All results can be reproduced using:

- The provided scripts  
- HuggingFace datasets  
- The evaluation setup described in this repository  


