"""
Model Evaluation and Visualization Module
Author: Sathish Kumar R (RA2411026050188)
Institution: SRM Institute of Science and Technology, Tiruchirappalli
Course: 21CSE377T - Information Retrieval
Assignment: CLA-2 Task 2
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set high-quality visualization style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11


def evaluate_all_models(pipelines: dict, X_test: np.ndarray, y_test: np.ndarray, training_summary: dict = None):
    """
    Evaluates all four trained pipeline models on the held-out test partition.
    Calculates accuracy, macro/weighted precision, recall, and F1-scores,
    generates confusion matrices, and writes reports.
    """
    labels = sorted(list(set(y_test)))
    metrics_records = []
    class_reports = {}
    confusion_matrices = {}

    print("\n" + "="*70)
    print("           MODEL EVALUATION ON TEST SET (20% HELD-OUT DATA)")
    print("="*70)

    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        report_dict = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
        report_text = classification_report(y_test, y_pred, labels=labels, zero_division=0)

        # Retrieve timing data if provided
        train_time = training_summary[name]['training_time_sec'] if training_summary and name in training_summary else 0.0
        pred_time = training_summary[name]['prediction_time_sec'] if training_summary and name in training_summary else 0.0
        pred_lat = training_summary[name]['prediction_latency_ms_per_sample'] if training_summary and name in training_summary else 0.0
        cv_mean = training_summary[name]['cv_mean_accuracy'] if training_summary and name in training_summary else 0.0
        cv_std = training_summary[name]['cv_std_accuracy'] if training_summary and name in training_summary else 0.0

        metrics_records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (Macro)": round(prec_macro, 4),
            "Recall (Macro)": round(rec_macro, 4),
            "F1-Score (Macro)": round(f1_macro, 4),
            "Precision (Weighted)": round(prec_weighted, 4),
            "Recall (Weighted)": round(rec_weighted, 4),
            "F1-Score (Weighted)": round(f1_weighted, 4),
            "5-Fold CV Mean": round(cv_mean, 4),
            "5-Fold CV Std": round(cv_std, 4),
            "Training Time (s)": round(train_time, 4),
            "Prediction Time (s)": round(pred_time, 4),
            "Latency (ms/doc)": round(pred_lat, 2)
        })

        class_reports[name] = report_dict
        confusion_matrices[name] = cm.tolist()

        print(f"\n>>> Model: {name}")
        print(f"    Accuracy: {acc*100:.2f}% | Macro F1: {f1_macro*100:.2f}% | Latency: {pred_lat:.2f} ms/doc")
        print("    Classification Report:")
        print(report_text)

    # Save metrics table to CSV
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df = metrics_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    metrics_csv_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n[OK] Metrics comparison table saved to: {metrics_csv_path}")

    # Save classification reports and confusion matrices to JSON
    with open(os.path.join(RESULTS_DIR, "classification_reports.json"), "w") as f:
        json.dump(class_reports, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "confusion_matrices.json"), "w") as f:
        json.dump({"labels": labels, "matrices": confusion_matrices}, f, indent=4)

    # Identify and save best model
    best_model_name = metrics_df.iloc[0]["Model"]
    best_pipeline = pipelines[best_model_name]
    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "best_model.joblib"))
    print(f"[OK] Champion Model identified: {best_model_name} (Saved to models/best_model.joblib)")

    return metrics_df, class_reports, confusion_matrices, labels


def generate_evaluation_visualizations(metrics_df: pd.DataFrame, confusion_matrices: dict, labels: list, df_raw: pd.DataFrame = None):
    """
    Renders and exports professional publication-grade visual charts:
    1. Overall Metrics Comparison Bar Chart
    2. Multi-Panel Confusion Matrices Heatmap
    3. Performance Radar Chart
    4. Training vs Prediction Speed Trade-off Scatter/Bar
    5. Class Distribution Bar & Donut Chart
    """
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 1. Bar Chart: Model Comparison
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    plot_df = metrics_df.melt(
        id_vars=['Model'],
        value_vars=['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
        var_name='Metric', value_name='Score'
    )
    sns.barplot(data=plot_df, x='Model', y='Score', hue='Metric', ax=ax, palette='Blues_r')
    ax.set_title("Performance Comparison Across Four Text Classification Algorithms\n(SRMIST CLA-2 Task 2 - BBC News Corpus)", fontsize=14, weight='bold', pad=15)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_xlabel("Classification Model", fontsize=12)
    ax.set_ylim(0.70, 1.02)
    for p in ax.patches:
        height = p.get_height()
        if height > 0.5:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=90, xytext=(0, 3),
                        textcoords='offset points')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    chart1_path = os.path.join(RESULTS_DIR, "model_comparison_bar.png")
    plt.savefig(chart1_path)
    plt.close()
    print(f"[OK] Saved chart: {chart1_path}")

    # 2. Confusion Matrices (4-panel subplot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    axes = axes.flatten()
    model_names = list(confusion_matrices.keys())

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        cm = np.array(confusion_matrices[name])
        # Compute normalized confusion matrix for annotation percentage
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels, cbar=False,
                    annot_kws={'size': 9, 'weight': 'bold'})
        ax.set_title(f"{name} Confusion Matrix", fontsize=12, weight='bold', pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Ground-Truth Label", fontsize=10)

    plt.suptitle("Confusion Matrix Comparison Across All Four Models\n(Test Set: 445 Samples)", fontsize=16, weight='bold', y=0.99)
    plt.tight_layout()
    chart2_path = os.path.join(RESULTS_DIR, "confusion_matrices_all.png")
    plt.savefig(chart2_path)
    plt.close()
    print(f"[OK] Saved chart: {chart2_path}")

    # 3. Training Time and Prediction Latency Trade-off
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    sns.barplot(data=metrics_df, x='Model', y='Training Time (s)', ax=ax1, palette='crest')
    ax1.set_title("Training Time (Seconds)", fontsize=12, weight='bold')
    ax1.set_ylabel("Time (seconds)", fontsize=11)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=20, ha='right')
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.3f}s", (p.get_x() + p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')

    sns.barplot(data=metrics_df, x='Model', y='Latency (ms/doc)', ax=ax2, palette='flare')
    ax2.set_title("Inference Latency per Document (Milliseconds)", fontsize=12, weight='bold')
    ax2.set_ylabel("Latency (ms / sample)", fontsize=11)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=20, ha='right')
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.2f}ms", (p.get_x() + p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')

    plt.suptitle("Computational Efficiency: Model Training vs Prediction Latency Benchmark", fontsize=14, weight='bold')
    plt.tight_layout()
    chart3_path = os.path.join(RESULTS_DIR, "training_prediction_time.png")
    plt.savefig(chart3_path)
    plt.close()
    print(f"[OK] Saved chart: {chart3_path}")

    # 4. Radar Chart for Multi-Metric Algorithm Comparison
    categories = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', '5-Fold CV Mean']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, row in metrics_df.iterrows():
        values = [row[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row['Model'], color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    plt.xticks(angles[:-1], categories, size=11, weight='bold')
    ax.set_ylim(0.75, 1.0)
    ax.set_title("Holistic Multi-Metric Radar Comparison\n(CLA-2 Task 2 Battle)", size=14, weight='bold', pad=25)
    plt.legend(loc='lower right', bbox_to_anchor=(1.25, 0.0), frameon=True)
    plt.tight_layout()
    chart4_path = os.path.join(RESULTS_DIR, "radar_chart.png")
    plt.savefig(chart4_path)
    plt.close()
    print(f"[OK] Saved chart: {chart4_path}")

    # 5. Class Distribution in BBC News Dataset
    if df_raw is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        counts = df_raw['category'].value_counts()
        colors_pie = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
        
        sns.barplot(x=counts.index, y=counts.values, ax=ax1, palette='mako')
        ax1.set_title("Article Frequency per Category", fontsize=12, weight='bold')
        ax1.set_ylabel("Number of Articles", fontsize=11)
        for p in ax1.patches:
            ax1.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2., p.get_height()),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 2), textcoords='offset points')

        ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140,
                colors=colors_pie, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax2.add_artist(centre_circle)
        ax2.set_title("Dataset Class Proportion", fontsize=12, weight='bold')

        plt.suptitle("BBC News Dataset Corpus Exploration (Total Records: 2,225)", fontsize=14, weight='bold')
        plt.tight_layout()
        chart5_path = os.path.join(RESULTS_DIR, "class_distribution.png")
        plt.savefig(chart5_path)
        plt.close()
        print(f"[OK] Saved chart: {chart5_path}")


if __name__ == "__main__":
    from src.model_training import load_and_preprocess_dataset, train_and_cross_validate_models
    df, _ = load_and_preprocess_dataset()
    X_train, X_test, y_train, y_test, pipelines, summary = train_and_cross_validate_models(df)
    metrics_df, reports, cms, labels = evaluate_all_models(pipelines, X_test, y_test, summary)
    generate_evaluation_visualizations(metrics_df, cms, labels, df)
    print("\nEvaluation pipeline executed successfully!")

