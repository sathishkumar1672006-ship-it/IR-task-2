"""
Model Training Pipeline Module for Information Retrieval & Text Classification
Author: Sathish Kumar R (RA2411026050188)
Institution: SRM Institute of Science and Technology, Tiruchirappalli
Course: 21CSE377T - Information Retrieval
Assignment: CLA-2 Task 2
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from src.preprocessing import TextPreprocessor

# Directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "bbc_news_clean.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_preprocess_dataset(data_path: str = DATA_PATH):
    """
    Loads dataset, applies text preprocessing, and encodes target labels.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"[1/5] Loading BBC dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(use_stemmer=False, use_lemmatizer=True)
    
    print("[2/5] Preprocessing news articles (lemmatization, stopwords, punctuation)...")
    # Clean text
    df['clean_text'] = df['text'].apply(lambda x: preprocessor.clean_text(x, method='lemmatize'))
    
    # Encode target labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['category'])
    
    # Save label encoder
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    
    return df, label_encoder


def get_model_pipelines(random_state: int = 42):
    """
    Defines Scikit-learn Pipelines combining TfidfVectorizer and the four required classifiers.
    Using Pipeline guarantees strict encapsulation: TF-IDF is fitted ONLY on the training split,
    preventing any target or feature leakage.
    """
    tfidf_config = {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'sublinear_tf': True,
        'min_df': 3,
        'norm': 'l2'
    }

    pipelines = {
        "Decision Tree": Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_config)),
            ('clf', DecisionTreeClassifier(criterion='gini', max_depth=50, random_state=random_state))
        ]),
        "K-Nearest Neighbours": Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_config)),
            ('clf', KNeighborsClassifier(n_neighbors=5, metric='cosine', weights='distance'))
        ]),
        "Support Vector Machine": Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_config)),
            ('clf', SVC(kernel='linear', C=1.0, probability=True, random_state=random_state))
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_config)),
            ('clf', MultinomialNB(alpha=0.1))
        ])
    }
    return pipelines


def train_and_cross_validate_models(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Splits data into stratified train/test partitions, executes 5-fold CV,
    trains all 4 models, tracks timing, and saves joblib artifacts.
    """
    X = np.array(df['clean_text'].tolist(), dtype=object)
    y = np.array(df['category'].tolist(), dtype=object)

    print(f"\n[3/5] Performing Stratified Train-Test Split (80% Train, 20% Test, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Testing samples:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    pipelines = get_model_pipelines(random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    training_summary = {}

    print("\n[4/5] Training Classifiers and Conducting 5-Fold Stratified Cross-Validation...")
    for model_name, pipeline in pipelines.items():
        print(f"\n---> Training: {model_name}...")
        
        # Cross validation on training partition only
        t0_cv = time.time()
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_time = time.time() - t0_cv
        
        # Fit on full training set
        t0_fit = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - t0_fit
        
        # Measure prediction latency on test set
        t0_pred = time.time()
        y_pred = pipeline.predict(X_test)
        pred_time = time.time() - t0_pred
        pred_latency_ms = (pred_time / len(X_test)) * 1000

        # Save trained pipeline artifact
        filename_key = model_name.lower().replace(" ", "_").replace("-", "_")
        model_save_path = os.path.join(MODELS_DIR, f"{filename_key}_pipeline.joblib")
        joblib.dump(pipeline, model_save_path)
        print(f"     Artifact saved to: {model_save_path}")

        training_summary[model_name] = {
            "cv_mean_accuracy": float(np.mean(cv_scores)),
            "cv_std_accuracy": float(np.std(cv_scores)),
            "cv_scores": [float(s) for s in cv_scores],
            "cv_time_sec": round(cv_time, 4),
            "training_time_sec": round(fit_time, 4),
            "prediction_time_sec": round(pred_time, 4),
            "prediction_latency_ms_per_sample": round(pred_latency_ms, 3),
            "artifact_path": model_save_path
        }
        print(f"     CV Accuracy: {np.mean(cv_scores)*100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)")
        print(f"     Fit Time: {fit_time:.4f}s | Pred Time: {pred_time:.4f}s ({pred_latency_ms:.2f} ms/doc)")

    # Save training metadata
    meta_path = os.path.join(MODELS_DIR, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(training_summary, f, indent=4)
    print(f"\n[5/5] Training metadata saved to: {meta_path}")

    return X_train, X_test, y_train, y_test, pipelines, training_summary


if __name__ == "__main__":
    df, le = load_and_preprocess_dataset()
    X_train, X_test, y_train, y_test, pipelines, summary = train_and_cross_validate_models(df)
    print("\nTraining completed successfully!")
