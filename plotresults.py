# Required Imports
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# GLOBAL STYLE (FONT SIZE FIX)
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})


def generate_all_evaluation_plots(
    judge_metrics_file="evaluation_metrics_qwenjudge.xlsx",
    model_metrics_file="evaluation_all_models.xlsx",
    base_output_dir="evaluation_outputs",
    humaneval_name="humaneval",
    humanevalnext_name="humanevalnext",
    dpi=300
):

    # Helper Functions
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    def sanitize_filename(name):
        return re.sub(r"[\\/:\*\?\"<>| ]+", "_", name)

    def clean_model_name(name):
        return name.split("/")[-1]

    def annotate_bars(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=11
            )

    # FIXED MODEL ORDER (NO Qwen2.5-3B)
    MODEL_ORDER = [
        "aiXcoder-7B",
        "starcoder2-7b",
        "CodeLlama-7b-Instruct-hf",
        "CodeLlama-7b-Python-hf",
        "deepseek-coder-6.7b-instruct",
        "Magicoder-S-DS-6.7B",
        "qwen2.5-coder-7b-instruct",
        "Mistral-7B-v0.3"
    ]

    # Output Directories
    judge_output_dir = os.path.join(base_output_dir, "judge_based_plots")
    model_output_dir = os.path.join(base_output_dir, "model_comparison_plots")
    ensure_dir(judge_output_dir)
    ensure_dir(model_output_dir)

    # PART 1: Judge-Based Metrics
    df = pd.read_excel(judge_metrics_file)
    df["dataset"] = df["dataset"].str.lower()

    df["test_model"] = df["test_model"].apply(clean_model_name)
    df["judge_model"] = df["judge_model"].apply(clean_model_name)

    agg_df = (
        df.groupby(["judge_model", "test_model", "dataset"])
        .agg(
            mean_avg_score=("avg_score", "mean"),
            mean_diversity=("diversity", "mean"),
            pct_max_score=("max_score", lambda x: (x == 1).mean() * 100),
            pct_min_score=("min_score", lambda x: (x == 0).mean() * 100),
        )
        .reset_index()
    )

    metrics = [
        ("mean_avg_score", "Mean of Average Scores"),
        ("mean_diversity", "Mean of Diversity Scores"),
        ("pct_max_score", "Percentage of Max Scores"),
        ("pct_min_score", "Percentage of Min Scores"),
    ]

    for judge in agg_df["judge_model"].unique():

        judge_df = agg_df[
            (agg_df["judge_model"] == judge) &
            (agg_df["test_model"].str.lower() != "vanilla")
        ]

        models = [
            m for m in MODEL_ORDER
            if m in judge_df["test_model"].unique() and m != judge
        ]

        x = np.arange(len(models))
        width = 0.35
        safe_judge = sanitize_filename(judge)

        for metric_col, metric_title in metrics:

            he_vals, hen_vals = [], []

            for m in models:
                he = judge_df[
                    (judge_df["test_model"] == m) &
                    (judge_df["dataset"] == humaneval_name)
                ][metric_col]

                hen = judge_df[
                    (judge_df["test_model"] == m) &
                    (judge_df["dataset"] == humanevalnext_name)
                ][metric_col]

                he_vals.append(he.values[0] if len(he) > 0 else 0)
                hen_vals.append(hen.values[0] if len(hen) > 0 else 0)

            fig, ax = plt.subplots(figsize=(12, 7))

            bars1 = ax.bar(x - width/2, he_vals, width, label="HumanEval")
            bars2 = ax.bar(x + width/2, hen_vals, width, label="HumanEvalNext")

            annotate_bars(bars1, ax)
            annotate_bars(bars2, ax)

            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=30, ha="right")

            ax.set_ylabel(metric_title)
            ax.set_xlabel("Test Model")
            ax.set_title(f"{metric_title} (Judge: {judge})")

            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.6)

            plt.tight_layout()
            plt.savefig(
                os.path.join(judge_output_dir, f"{safe_judge}_{metric_col}.png"),
                dpi=dpi,
                bbox_inches="tight"
            )
            plt.close()

    # MEAN AVG SCORE (FINAL CONSISTENT PLOT)
    
    for judge in agg_df["judge_model"].unique():

        judge_df = agg_df[
            (agg_df["judge_model"] == judge) &
            (agg_df["test_model"].str.lower() != "vanilla")
        ]

        models = [
            m for m in MODEL_ORDER
            if m in judge_df["test_model"].unique() and m != judge
        ]

        x = np.arange(len(models))
        width = 0.35

        he_vals, hen_vals = [], []

        for m in models:
            he = judge_df[
                (judge_df["test_model"] == m) &
                (judge_df["dataset"] == humaneval_name)
            ]["mean_avg_score"]

            hen = judge_df[
                (judge_df["test_model"] == m) &
                (judge_df["dataset"] == humanevalnext_name)
            ]["mean_avg_score"]

            he_vals.append(he.values[0] if len(he) > 0 else 0)
            hen_vals.append(hen.values[0] if len(hen) > 0 else 0)

        fig, ax = plt.subplots(figsize=(12, 7))

        bars1 = ax.bar(x - width/2, he_vals, width, label="HumanEval")
        bars2 = ax.bar(x + width/2, hen_vals, width, label="HumanEvalNext")

        annotate_bars(bars1, ax)
        annotate_bars(bars2, ax)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")

        ax.set_ylabel("Mean of Average Scores")
        ax.set_xlabel("Test Model")
        ax.set_title(f"Mean Average Score (Judge: {judge})")

        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()

        safe_judge = sanitize_filename(judge)
        plt.savefig(
            os.path.join(judge_output_dir, f"{safe_judge}_mean_avg_score.png"),
            dpi=dpi,
            bbox_inches="tight"
        )
        plt.close()

    df = pd.read_excel(model_metrics_file)

    df["model_clean"] = df["model"].apply(clean_model_name)

    x_order = [
        m for m in MODEL_ORDER if m in df["model_clean"].unique()
    ]

    def plot_metric(metric_name):

        metric_df = df[df["evaluation_metric"] == metric_name]

        pivot = metric_df.pivot(
            index="model_clean",
            columns="dataset",
            values="result"
        ).reindex(x_order).fillna(0)

        x = np.arange(len(pivot.index))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 7))

        bars1 = ax.bar(x - width/2, pivot[humaneval_name], width, label="HumanEval")
        bars2 = ax.bar(x + width/2, pivot[humanevalnext_name], width, label="HumanEvalNext")

        annotate_bars(bars1, ax)
        annotate_bars(bars2, ax)

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=30, ha="right")

        ax.set_ylabel(metric_name)
        ax.set_xlabel("Models")
        ax.set_title(f"{metric_name} Comparison")

        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(
            os.path.join(model_output_dir, f"{metric_name}.png"),
            dpi=dpi,
            bbox_inches="tight"
        )
        plt.close()

    for metric in ["pass@1", "pass@5", "avg_unique_solution_rate", "bleu"]:
        plot_metric(metric)


# Entry Point
if __name__ == "__main__":
    generate_all_evaluation_plots()
