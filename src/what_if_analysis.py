"""
What-If Experimental Analysis Module
Author: Sathish Kumar R (RA2411026050188)
Institution: SRM Institute of Science and Technology, Tiruchirappalli
Course: 21CSE377T - Information Retrieval
Assignment: CLA-2 Task 2
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "bbc_news_clean.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_knn_sensitivity_experiment(X_train, X_test, y_train, y_test):
    """
    Experiment 1: Systematically varies K-Nearest Neighbours k in [1, 3, 5, 7, 9, 11, 15, 21, 31]
    and compares Cosine vs Euclidean distance metrics.
    """
    print("\n--- Running What-If Experiment 1: KNN Neighborhood Size (k) and Distance Metric ---")
    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    results = []
    for metric in ['cosine', 'euclidean']:
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')
            knn.fit(X_train_vec, y_train)
            y_pred = knn.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            results.append({
                'Experiment': 'KNN Sensitivity',
                'Parameter_k': k,
                'Metric': metric,
                'Accuracy': round(acc, 4),
                'Accuracy_Pct': round(acc * 100, 2),
                'F1_Macro': round(f1, 4)
            })
            print(f"  Metric: {metric:9s} | k={k:2d} -> Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")

    df_knn = pd.DataFrame(results)
    df_knn.to_csv(os.path.join(RESULTS_DIR, "what_if_knn.csv"), index=False)
    return df_knn


def run_tfidf_feature_capacity_experiment(X_train, X_test, y_train, y_test):
    """
    Experiment 2: Evaluates impact of TF-IDF vocabulary capacity (max_features)
    from 250 up to 10,000 and unlimited features on SVM and Naive Bayes.
    """
    print("\n--- Running What-If Experiment 2: TF-IDF Vocabulary Capacity (max_features) ---")
    feature_capacities = [250, 500, 1000, 2000, 3000, 5000, 8000, 12000, None]
    results = []

    for max_feat in feature_capacities:
        feat_label = "Unlimited" if max_feat is None else str(max_feat)
        tfidf = TfidfVectorizer(max_features=max_feat, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        X_tr = tfidf.fit_transform(X_train)
        X_te = tfidf.transform(X_test)
        actual_features = X_tr.shape[1]

        # SVM evaluation
        svm = SVC(kernel='linear', C=1.0, random_state=42)
        svm.fit(X_tr, y_train)
        svm_acc = accuracy_score(y_test, svm.predict(X_te))
        svm_f1 = f1_score(y_test, svm.predict(X_te), average='macro')

        # Naive Bayes evaluation
        nb = MultinomialNB(alpha=0.1)
        nb.fit(X_tr, y_train)
        nb_acc = accuracy_score(y_test, nb.predict(X_te))
        nb_f1 = f1_score(y_test, nb.predict(X_te), average='macro')

        results.append({
            'max_features_setting': feat_label,
            'actual_vocab_size': actual_features,
            'SVM_Accuracy': round(svm_acc, 4),
            'SVM_F1': round(svm_f1, 4),
            'NB_Accuracy': round(nb_acc, 4),
            'NB_F1': round(nb_f1, 4)
        })
        print(f"  max_features={feat_label:9s} (Vocab: {actual_features:5d}) -> SVM: {svm_acc*100:.2f}% | NB: {nb_acc*100:.2f}%")

    df_feat = pd.DataFrame(results)
    df_feat.to_csv(os.path.join(RESULTS_DIR, "what_if_features.csv"), index=False)
    return df_feat


def run_ngram_range_experiment(X_train, X_test, y_train, y_test):
    """
    Experiment 3: Tests N-Gram range effect (Unigrams, Bigrams, Trigrams) across all 4 algorithms.
    """
    print("\n--- Running What-If Experiment 3: N-Gram Range Impact ---")
    ngram_configs = {
        'Unigrams (1,1)': (1, 1),
        'Unigram + Bigram (1,2)': (1, 2),
        'Unigram + Bigram + Trigram (1,3)': (1, 3)
    }

    results = []
    for ngram_name, ngram_tuple in ngram_configs.items():
        tfidf = TfidfVectorizer(ngram_range=ngram_tuple, max_features=5000, sublinear_tf=True, min_df=3)
        X_tr = tfidf.fit_transform(X_train)
        X_te = tfidf.transform(X_test)

        clfs = {
            'Decision Tree': DecisionTreeClassifier(max_depth=50, random_state=42),
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, metric='cosine', weights='distance'),
            'SVM (Linear)': SVC(kernel='linear', C=1.0, random_state=42),
            'Naive Bayes': MultinomialNB(alpha=0.1)
        }

        for model_name, clf in clfs.items():
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_te)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            results.append({
                'N-Gram Configuration': ngram_name,
                'Model': model_name,
                'Accuracy': round(acc, 4),
                'F1_Macro': round(f1, 4)
            })
            print(f"  {ngram_name:32s} | {model_name:15s} -> Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")

    df_ngram = pd.DataFrame(results)
    df_ngram.to_csv(os.path.join(RESULTS_DIR, "what_if_ngram.csv"), index=False)
    return df_ngram


def plot_what_if_experiments(df_knn, df_feat, df_ngram):
    """
    Produces publication-grade multi-panel figure visualizing the 3 What-If experiments.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Panel 1: KNN Sensitivity
    ax1 = axes[0]
    sns.lineplot(data=df_knn, x='Parameter_k', y='Accuracy', hue='Metric', marker='o', linewidth=2.5, ax=ax1, palette=['#1f77b4', '#e74c3c'])
    ax1.set_title("Experiment 1: KNN k Sensitivity & Distance Metric", fontsize=12, weight='bold')
    ax1.set_xlabel("Neighborhood Size (k)", fontsize=11)
    ax1.set_ylabel("Accuracy Score", fontsize=11)
    ax1.set_ylim(0.70, 1.0)
    ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.7, label='Optimal k=5')
    ax1.legend(title="Distance Metric", loc='lower right')

    # Panel 2: TF-IDF Vocabulary Capacity
    ax2 = axes[1]
    # For plotting on x-axis, replace None with label
    x_labels = df_feat['max_features_setting'].tolist()
    ax2.plot(range(len(x_labels)), df_feat['SVM_Accuracy'], marker='s', linewidth=2.5, color='#2ca02c', label='SVM (Linear)')
    ax2.plot(range(len(x_labels)), df_feat['NB_Accuracy'], marker='^', linewidth=2.5, color='#ff7f0e', label='Naive Bayes')
    ax2.set_xticks(range(len(x_labels)))
    ax2.set_xticklabels(x_labels, rotation=35, ha='right')
    ax2.set_title("Experiment 2: Vocabulary Size (max_features)", fontsize=12, weight='bold')
    ax2.set_xlabel("TF-IDF max_features Setting", fontsize=11)
    ax2.set_ylabel("Accuracy Score", fontsize=11)
    ax2.set_ylim(0.85, 1.0)
    ax2.legend(loc='lower right')

    # Panel 3: N-Gram Impact
    ax3 = axes[2]
    sns.barplot(data=df_ngram, x='Model', y='Accuracy', hue='N-Gram Configuration', ax=ax3, palette='magma')
    ax3.set_title("Experiment 3: N-Gram Range Comparison", fontsize=12, weight='bold')
    ax3.set_xlabel("Classifier", fontsize=11)
    ax3.set_ylabel("Accuracy Score", fontsize=11)
    ax3.set_ylim(0.75, 1.0)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=20, ha='right')
    ax3.legend(title="N-Gram Range", loc='lower right')

    plt.suptitle("Comprehensive What-If Empirical Analysis: Parameter & Feature Sensitivity\n(CLA-2 Task 2 Academic Investigation)", fontsize=15, weight='bold', y=1.02)
    plt.tight_layout()
    whatif_path = os.path.join(RESULTS_DIR, "what_if_analysis.png")
    plt.savefig(whatif_path, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] What-If Analysis multi-panel visualization saved to: {whatif_path}")


def execute_what_if_pipeline():
    from src.model_training import load_and_preprocess_dataset
    df, _ = load_and_preprocess_dataset()
    X = np.array(df['clean_text'].tolist(), dtype=object)
    y = np.array(df['category'].tolist(), dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    df_knn = run_knn_sensitivity_experiment(X_train, X_test, y_train, y_test)
    df_feat = run_tfidf_feature_capacity_experiment(X_train, X_test, y_train, y_test)
    df_ngram = run_ngram_range_experiment(X_train, X_test, y_train, y_test)
    plot_what_if_experiments(df_knn, df_feat, df_ngram)
    print("\n[OK] All What-If Experiments completed and recorded!")


if __name__ == "__main__":
    execute_what_if_pipeline()
