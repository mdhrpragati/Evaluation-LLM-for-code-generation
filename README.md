
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
