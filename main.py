"""
Main Execution Script for Information Retrieval & Text Classification Battle
Student: Sathish Kumar R (Register Number: RA2411026050188)
Institution: SRM Institute of Science and Technology, Tiruchirappalli
Course: 21CSE377T - Information Retrieval
Assignment: CLA-2 Task 2 - TEXT CLASSIFICATION BATTLE: COMPARE, ANALYZE AND DEFEND
"""

import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import get_default_preprocessor
from src.model_training import load_and_preprocess_dataset, train_and_cross_validate_models
from src.evaluation import evaluate_all_models, generate_evaluation_visualizations
from src.what_if_analysis import execute_what_if_pipeline
from src.prediction import NewsPredictor


def print_banner():
    banner = """
================================================================================
          SRM INSTITUTE OF SCIENCE AND TECHNOLOGY, TIRUCHIRAPPALLI
             DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
               21CSE377T - INFORMATION RETRIEVAL | CLA-2 TASK 2
--------------------------------------------------------------------------------
         TEXT CLASSIFICATION BATTLE: COMPARE, ANALYZE AND DEFEND
                 NEWS CATEGORY CLASSIFICATION USING ML
--------------------------------------------------------------------------------
 Student Name : Sathish Kumar R
 Register No  : RA2411026050188
 Max Marks    : 30
 Dataset      : BBC News Corpus (2,225 Articles, 5 Categories)
 Classifiers  : Decision Tree | KNN | SVM | Multinomial Naive Bayes
================================================================================
    """
    print(banner)


def run_full_pipeline():
    start_time = time.time()
    print_banner()

    # Step 1: Preprocessing & Data Loading
    print("\n>>> STAGE 1: DATA INGESTION & TEXT PREPROCESSING")
    df, label_encoder = load_and_preprocess_dataset()
    print(f"    Loaded and preprocessed {len(df)} articles across categories: {list(label_encoder.classes_)}")

    # Step 2: Training & Cross-Validation
    print("\n>>> STAGE 2: PIPELINE TRAINING & 5-FOLD CROSS VALIDATION")
    X_train, X_test, y_train, y_test, pipelines, summary = train_and_cross_validate_models(df)

    # Step 3: Evaluation & Metric Compilation
    print("\n>>> STAGE 3: HELD-OUT TEST EVALUATION & METRIC CALCULATION")
    metrics_df, class_reports, confusion_matrices, labels = evaluate_all_models(
        pipelines, X_test, y_test, summary
    )

    # Step 4: Visualizations
    print("\n>>> STAGE 4: GENERATING PUBLICATION-GRADE VISUALIZATIONS")
    generate_evaluation_visualizations(metrics_df, confusion_matrices, labels, df)

    # Step 5: What-If Sensitivity Experiments
    print("\n>>> STAGE 5: EXECUTING WHAT-IF EMPIRICAL EXPERIMENTS")
    execute_what_if_pipeline()

    # Step 6: Sample Live Inference Demonstration
    print("\n>>> STAGE 6: LIVE INFERENCE VERIFICATION")
    predictor = NewsPredictor(model_name="best_model")
    samples = [
        "Premier League leaders Arsenal defeated Manchester City with a stoppage-time goal from Bukayo Saka.",
        "Wall Street stocks rallied following Federal Reserve interest rate cut hints and quarterly revenue surges.",
        "Google and Microsoft introduced generative AI models capable of autonomous multi-modal reasoning and code generation.",
        "Prime Minister Tony Blair defended national security policies during an intense House of Commons debate.",
        "The British film Academy honored director Christopher Nolan with Best Picture BAFTA at Royal Opera House."
    ]
    df_preds = predictor.predict_batch(samples)
    print("\nSample Real-World Inferences:")
    print(df_preds.to_string(index=False))

    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print(f" PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("="*80)
    print(" Outputs Generated:")
    print("   1. Trained Models     -> models/ (*.joblib)")
    print("   2. Metric Results     -> results/metrics_comparison.csv")
    print("   3. Visual Plots       -> results/*.png")
    print("   4. What-If Records    -> results/what_if_*.csv")
    print(" Next Step:")
    print("   Launch Streamlit Dashboard: streamlit run app.py")
    print("="*80)


if __name__ == "__main__":
    run_full_pipeline()

