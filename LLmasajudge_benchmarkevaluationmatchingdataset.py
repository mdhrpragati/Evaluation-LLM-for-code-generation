import os
import re
import json
import time
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


user_tmp = f"{Path.home()}/hf_cache"
Path(user_tmp).mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = user_tmp
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


print(f" HPC cache directory set to: {user_tmp}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

def safe_load_dataset(name, split=None, retries=3):
    for i in range(retries):
        try:
            ds = load_dataset(name)
            return ds[split] if split else ds
        except Exception as e:
            print(f"⚠️ Dataset load failed ({name}) attempt {i+1}: {e}")
            time.sleep(5)

    raise RuntimeError(f"Failed to load dataset: {name}")

def load_model(name):
    print(f" Loading model: {name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            cache_dir=os.environ["HF_HOME"]
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=os.environ["HF_HOME"],
            low_cpu_mem_usage=True
        ).eval()

        return model, tokenizer

    except Exception as e:
        print(f" Failed to load {name}: {e}")
        return None, None


def extract_json(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    return matches[-1] if matches else None


def clamp_scores(result):
    keys = [
        "correct_tests_score",
        "coverage_score",
        "solution_score",
        "clarity_score"
    ]

    for k in keys:
        if k in result:
            try:
                result[k] = int(result[k])
                result[k] = max(1, min(5, result[k]))
            except:
                result[k] = 1

    return result

def audit_with_rubric(model, tokenizer, prompt, tests, solution):

    audit_prompt = f"""
You are a strict code evaluation system.

Evaluate benchmark quality.

Problem:
{prompt}

Tests:
{tests}

Solution:
{solution}

RULES:
- Output ONLY JSON
- Scores MUST be integers 1-5
- NEVER use percentages
- NEVER exceed 5

Scale:
1 = very poor
2 = poor
3 = average
4 = good
5 = excellent

Return:

{{
  "correct_tests_score": int,
  "coverage_score": int,
  "solution_score": int,
  "clarity_score": int,
  "summary": "short explanation"
}}
"""

    inputs = tokenizer(
        audit_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        json_str = extract_json(text)
        if not json_str:
            raise ValueError("No JSON found")

        result = json.loads(json_str)
        return clamp_scores(result)

    except:
        return {
            "correct_tests_score": 1,
            "coverage_score": 1,
            "solution_score": 1,
            "clarity_score": 1,
            "summary": "parsing failed"
        }


def get_func_name(prompt):
    match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", prompt)
    return match.group(1) if match else None


judge_models_local = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
]


def run_audit():

    print(" Loading datasets...")

    humaneval = safe_load_dataset("openai_humaneval", "test")
    humanevalnext = safe_load_dataset("AISE-TUDelft/HumanEvalNext", "train")

    use_task_id = "task_id" in humaneval.column_names and "task_id" in humanevalnext.column_names

    if use_task_id:
        he_map = {x["task_id"]: x for x in humaneval}
        hn_map = {x["task_id"]: x for x in humanevalnext}
        common_keys = set(he_map) & set(hn_map)
    else:
        he_map = {get_func_name(x["prompt"]): x for x in humaneval}
        hn_map = {get_func_name(x["prompt"]): x for x in humanevalnext}

        he_map = {k: v for k, v in he_map.items() if k}
        hn_map = {k: v for k, v in hn_map.items() if k}
        common_keys = set(he_map) & set(hn_map)

    print(f" Matched samples: {len(common_keys)}")

    
    for judge_model_name in judge_models_local:

        model, tokenizer = load_model(judge_model_name)

        if model is None:
            print(f" Skipping {judge_model_name}")
            continue

        results = []

        for key in tqdm(common_keys, desc=f"Evaluating {judge_model_name}"):

            he_item = he_map[key]
            hn_item = hn_map[key]

            # HumanEval
            res_he = audit_with_rubric(
                model,
                tokenizer,
                he_item["prompt"],
                he_item.get("test", ""),
                he_item.get("canonical_solution", "")
            )

            results.append({
                "judge_model": judge_model_name,
                "dataset": "HumanEval",
                "task_id": key,
                **res_he
            })

            # HumanEvalNext
            res_hn = audit_with_rubric(
                model,
                tokenizer,
                hn_item["prompt"],
                hn_item.get("test", ""),
                hn_item.get("canonical_solution", "")
            )

            results.append({
                "judge_model": judge_model_name,
                "dataset": "HumanEvalNext",
                "task_id": key,
                **res_hn
            })

   
        df = pd.DataFrame(results)

        output_file = f"audit_resultsfirst_{judge_model_name.replace('/', '_')}.xlsx"
        df.to_excel(output_file, index=False)

        print(f"\n Saved: {output_file}")

        print("\n Averages:")
        print(df.groupby("dataset")[[
            "correct_tests_score",
            "coverage_score",
            "solution_score",
            "clarity_score"
        ]].mean())

        # cleanup
        del model, tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_audit()