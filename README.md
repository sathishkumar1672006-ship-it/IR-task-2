# NEWS CATEGORY CLASSIFICATION USING MACHINE LEARNING
## Text Classification Battle: Compare, Analyze and Defend
### Information Retrieval (21CSE377T) – CLA-2 Task 2 Project

---

### Student Credentials
- **Student Name:** Sathish Kumar R
- **Register Number:** RA2411026050188
- **Institution:** SRM Institute of Science and Technology, Tiruchirappalli Campus
- **Degree & Branch:** B.Tech. Computer Science and Engineering
- **Course Code & Title:** 21CSE377T – Information Retrieval
- **Assignment:** Continuous Learning Assessment – 2 (Task 2)
- **Maximum Marks:** 30

---

## 1. Project Overview & Problem Statement
Automated text categorization is a core challenge in Information Retrieval (IR). When search engines, news aggregators, and digital publishing houses ingest tens of thousands of unlabelled documents daily, automated categorization allows partitioning of inverted indices, fast topical query routing, and personalized recommendation feeds.

In this project, an end-to-end Machine Learning and Information Retrieval system is built to classify news articles into 5 topical domains:
1. **Business**
2. **Entertainment**
3. **Politics**
4. **Sport**
5. **Tech**

Four foundational machine learning classifiers representing contrasting theoretical paradigms are trained, benchmarked, and defended:
- **Decision Tree Classifier (CART)**
- **K-Nearest Neighbours (KNN, Cosine metric)**
- **Support Vector Machine (Linear SVM)**
- **Multinomial Naive Bayes (MNB)**

---

## 2. Directory Structure

```text
d:\IR task2 sathish\
├── app.py                      # Interactive Streamlit Web Application (8 modules)
├── main.py                     # Master CLI pipeline orchestrator
├── requirements.txt            # Python dependencies with exact versions
├── .gitignore                  # Git ignore rules for Python, models & checkpoints
├── README.md                   # Complete documentation and execution guide
├── execute_notebook.py         # Script to run and populate Jupyter Notebook cells
├── generate_presentation.py    # Script generating 15-slide PowerPoint deck
├── generate_academic_report.py # Script generating formal Word report (.docx)
│
├── dataset/                    # Dataset directory
│   ├── bbc_news.csv            # Raw BBC News dataset (2,225 records)
│   ├── bbc_news_clean.csv      # Clean UTF-8 encoded dataset (text, category)
│   └── DATASET_INFO.md         # Comprehensive provenance & data audit report
│
├── src/                        # Modular source code
│   ├── __init__.py             # Package initializer
│   ├── preprocessing.py        # TextPreprocessor (lowercasing, regex, stopwords, lemmatization)
│   ├── model_training.py       # Pipeline construction, stratified split & 5-fold CV
│   ├── evaluation.py           # Metric calculation (Accuracy, F1, latency) & chart plotting
│   ├── what_if_analysis.py     # Parametric sensitivity experiments (k, max_features, n-grams)
│   └── prediction.py           # Real-time inference engine with keyword explainability
│
├── models/                     # Serialized model artifacts (Joblib)
│   ├── best_model.joblib
│   ├── support_vector_machine_pipeline.joblib
│   ├── multinomial_naive_bayes_pipeline.joblib
│   ├── k_nearest_neighbours_pipeline.joblib
│   ├── decision_tree_pipeline.joblib
│   ├── label_encoder.joblib
│   └── training_metadata.json
│
├── results/                    # Quantitative evaluation outputs & plots
│   ├── metrics_comparison.csv  # Official performance leaderboard
│   ├── classification_reports.json # Full per-class precision, recall, F1, support
│   ├── confusion_matrices.json # Raw and normalized confusion matrices
│   ├── what_if_knn.csv         # Experiment 1: k sensitivity (1 to 31)
│   ├── what_if_features.csv    # Experiment 2: vocabulary capacity (250 to Unlimited)
│   ├── what_if_ngram.csv       # Experiment 3: N-gram range comparisons
│   ├── model_comparison_bar.png# Comparative bar chart across all 4 models
│   ├── confusion_matrices_all.png # 4-Panel confusion matrix heatmaps
│   ├── radar_chart.png         # Multi-metric polar radar profile
│   ├── training_prediction_time.png # Training time vs inference latency trade-offs
│   ├── what_if_analysis.png    # Multi-panel What-If experimental curves
│   └── class_distribution.png  # Class balance & word count distribution
│
├── notebooks/                  # Executable Jupyter Notebooks
│   └── News_Classification_Battle.ipynb # Fully executed academic notebook
│
├── academic_report/            # Formal academic research papers
│   ├── ACADEMIC_REPORT.docx    # Professional Word report with embedded plots
│   └── ACADEMIC_REPORT.md      # Full Markdown academic report
│
├── presentation/               # Oral technical defense presentation
│   └── News_Category_Classification_Battle.pptx # 15 slides with speaker notes
│
└── viva/                       # Oral defense preparation
    └── VIVA_QUESTIONS.md       # 20+ comprehensive technical Q&As
```

---

## 3. Dataset Description & Provenance
- **Dataset Name:** BBC News Classification Corpus (2004–2005)
- **Original Research Source:** D. Greene and P. Cunningham. *"Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering"*, Proceedings of the 23rd International Conference on Machine Learning (ICML 2006).
- **Official University Archive:** [University College Dublin / Insight Centre](http://mlg.ucd.ie/datasets/bbc.html)
- **Total Records:** 2,225 genuine news articles (No synthetic/fabricated data).
- **Missing Values:** Exactly **0** missing values.
- **Duplicate Records:** 98 identical wire syndication records detected and documented.

### Class Distribution

| Category | Articles | Proportion | Mean Words | Median Words | Scope |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sport** | 511 | 22.97% | 329.26 | 288.0 | Football, rugby, athletics, tennis, cricket |
| **Business** | 510 | 22.92% | 328.88 | 297.0 | Stock markets, corporate profits, economy, banking |
| **Politics** | 417 | 18.74% | 453.97 | 439.0 | UK Parliament, Blair government, elections, taxes |
| **Tech** | 401 | 18.02% | 502.70 | 447.0 | Mobile phones, broadband, cyber security, gadgets |
| **Entertainment** | 386 | 17.35% | 330.62 | 262.5 | Film awards (Oscars, BAFTAs), music, festivals |

---

## 4. Text Preprocessing & TF-IDF Vectorization

### 4.1 Linguistic Preprocessing Pipeline
Raw text is processed through 5 sequential transformations in `src/preprocessing.py`:
1. **Lowercase Conversion:** Conflates capitalized and lowercase terms (`"Market"` vs `"market"`).
2. **Regex Symbol & Numeric Stripping:** Removes URLs, HTML tags, punctuation, currency symbols (`£`), and numbers.
3. **Tokenization:** Splits text into alphabetic tokens.
4. **Stop-Word Removal:** Prunes high-frequency English particles (`"the"`, `"is"`, `"which"`, `"from"`).
5. **WordNet Lemmatization:** Reduces inflected forms to root lemmas (`"shares"`, `"sharing"` $\rightarrow$ `"share"`).

### 4.2 TF-IDF Mathematical Representation
Documents are converted to 5,000-dimensional numerical vectors using:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

- **Sublinear Term Frequency:** $\text{sublinear\_TF}(t, d) = 1 + \log(f_{t, d})$ for $f_{t, d} > 0$.
- **Smooth IDF:** $\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$.
- **L2 Hypersphere Normalization:** $\|\mathbf{v}\|_2 = 1$, neutralizing document length discrepancies.
- **N-Gram Range:** `ngram_range=(1, 2)` (captures both unigrams and essential bigrams such as `"prime minister"`, `"interest rate"`, `"wall street"`).

### 4.3 Data Leakage Prevention
Feature extraction is enclosed within Scikit-Learn `Pipeline` objects. The vocabulary and IDF weights are computed **strictly on the 80% training partition ($X_{train}$)**. During test evaluation, $X_{test}$ is transformed using the frozen training vocabulary, strictly preventing information leakage.

---

## 5. Official Classification Battle Leaderboard (Actual Results)

Evaluated on the held-out test split of 445 articles (20% stratified test partition):

| Classifier | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 | 5-Fold CV Mean | 5-Fold CV Std | Train Time (s) | Latency (ms/doc) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multinomial Naive Bayes** | **97.98%** | **0.9791** | **0.9813** | **0.9799** | **0.9797** | 96.97% | ± 0.0074 | **0.8440 s** | **0.16 ms** |
| **Support Vector Machine** | **97.98%** | **0.9801** | **0.9796** | **0.9796** | **0.9797** | **98.48%** | **± 0.0029** | 12.5751 s | 1.18 ms |
| **K-Nearest Neighbours** | **97.08%** | **0.9708** | **0.9712** | **0.9708** | **0.9709** | 96.12% | ± 0.0113 | 0.8327 s | 0.21 ms |
| **Decision Tree (CART)** | **85.17%** | **0.8534** | **0.8521** | **0.8524** | **0.8521** | 83.54% | ± 0.0159 | 1.0456 s | 0.16 ms |

### Student Technical Defense Summary
- **Why Linear SVM & Naive Bayes Tied at 97.98%:** Linear SVM maximizes the margin hyperplane separating classes in 5,000-dimensional space, yielding the highest cross-validation stability (**98.48%**). Naive Bayes computes linear log-odds decision surfaces that preserve rank ordering despite word independence violations.
- **Why Naive Bayes is the Deployment Winner:** At **0.16 ms per article**, Naive Bayes runs 7.4× faster than calibrated SVM, allowing a single server core to process over 6,000 news articles per second.
- **Why Decision Trees Lagged (85.17%):** Single-word orthogonal axis-aligned cuts cannot capture semantic topics distributed across hundreds of correlated terms.

---

## 6. What-If Empirical Sensitivity Analysis

Three real experiments were executed in `src/what_if_analysis.py`:
1. **KNN Neighborhood Size ($k$) and Metric:** Tested $k \in [1, 31]$. Accuracy peaked at $k=3$ (97.98%) and stabilized at $k=5$ (97.08%). Distance-weighted voting ($w = 1/d$) cushioned larger $k$ values. Cosine distance proved strictly superior to Euclidean distance.
2. **TF-IDF Vocabulary Capacity (`max_features`):** Tested from 250 to Unlimited (60,659 terms). Accuracy rose sharply from 250 (94.38%) to 3,000 features. Beyond 5,000 features, performance reached an asymptote (97.98% for both 5,000 and 60,659 features), confirming that `max_features=5000` is optimal.
3. **N-Gram Lexical Span:** Adding bigrams `(1, 2)` improved accuracy over unigrams `(1, 1)` by +0.45% in SVM and +0.68% in KNN by capturing phrasal collocations (`"interest rate"`, `"prime minister"`). Trigrams yielded diminishing returns.

---

## 7. Windows Command Prompt (CMD) Execution Guide

Follow these exact commands in Windows Command Prompt (`cmd.exe`):

### Step 1: Open CMD and Navigate to Workspace
```cmd
cd /d "d:\IR task2 sathish"
```

### Step 2: Install Python Dependencies
```cmd
pip install -r requirements.txt
```

### Step 3: Run the End-to-End Pipeline
Executes data preprocessing, trains all 4 models, runs 5-fold cross-validation, generates metrics table, plots PNG visualizations, runs what-if experiments, and verifies test inferences:
```cmd
python main.py
```

### Step 4: Launch the Streamlit Web Application
```cmd
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore the live dashboard.

### Step 5: (Optional) Re-generate Academic Report & Presentation
```cmd
python generate_academic_report.py
python generate_presentation.py
```

---

## 8. Student Final Submission & Video Demonstration Checklist

| Item | Requirement | Status | File Location |
| :--- | :--- | :--- | :--- |
| **Source Code** | Modular Python modules | Complete | `src/*.py`, `main.py` |
| **Web UI** | 8-module Streamlit app | Complete | `app.py` |
| **Dataset** | Authentic BBC News (2,225 records) | Verified | `dataset/bbc_news_clean.csv` |
| **Trained Models** | Serialized Joblib pipelines | Generated | `models/*.joblib` |
| **Results & Charts**| CSV metrics & PNG graphs | Generated | `results/` |
| **Jupyter Notebook**| Fully executed notebook with outputs | Complete | `notebooks/News_Classification_Battle.ipynb` |
| **Academic Report**| Formal Word document with title page | Complete | `academic_report/ACADEMIC_REPORT.docx` |
| **PowerPoint Deck** | 15 slides with speaker notes | Complete | `presentation/News_Category_Classification_Battle.pptx` |
| **Viva Questions** | 20+ comprehensive technical Q&As | Complete | `viva/VIVA_QUESTIONS.md` |

### Manual Actions Required by Student:
1. **Video Recording:** Launch Streamlit (`streamlit run app.py`), record a 3-5 minute video demonstrating:
   - Home tab showing your student credentials (Sathish Kumar R, RA2411026050188)
   - Preprocessing Lab demonstrating token reduction
   - Four-Model Battle Arena showing the leaderboard
   - Live News Classifier predicting custom news articles
2. **GitHub Submission:** Initialize git repo, commit project files, and push to GitHub (using `.gitignore`).
3. **Report Submission:** Submit `ACADEMIC_REPORT.docx` and `News_Category_Classification_Battle.pptx` on the university portal.

