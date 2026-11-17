# Comprehensive Viva-Voce Technical Defense & Examination Guide

**Course:** 21CSE377T – Information Retrieval  
**Assignment:** CLA-2 Task 2 – Text Classification Battle: Compare, Analyze and Defend  
**Student Name:** Sathish Kumar R  
**Register Number:** RA2411026050188  
**Institution:** SRM Institute of Science and Technology, Tiruchirappalli Campus  
**Project Title:** News Category Classification Using Machine Learning  

---

## Section A: Information Retrieval, Text Preprocessing & TF-IDF

### Q1. What is the core problem of Information Retrieval addressed in this project?
**Answer:**  
In Information Retrieval (IR), automated text categorization addresses the problem of indexing, routing, and organizing unstructured natural language documents into predefined thematic categories. Without automated classification, retrieval systems must execute costly full-text scans. By mapping text into numerical vector spaces and categorizing them into distinct topical domains (*Business, Entertainment, Politics, Sport, Tech*), search engines and digital news aggregators can partition indices, optimize query routing, and deliver relevant documents with sub-millisecond latency.

### Q2. Walk through each step of your text preprocessing pipeline. Why is each step essential?
**Answer:**  
Our preprocessing pipeline implements five sequential operations:
1. **Lowercase Conversion:** Maps all characters to lowercase to prevent identical lexical tokens (e.g., `"Apple"` vs `"apple"`) from creating separate feature dimensions.
2. **Regex Filtering:** Strips URLs, HTML markup, numerical digits, and punctuation symbols, isolating pure alphabetic semantic content.
3. **Tokenization:** Segments continuous document text into discrete word tokens.
4. **Stop-Word Removal:** Eliminates high-frequency grammatical functional words (e.g., `"the"`, `"is"`, `"which"`, `"at"`) that carry high collection frequency but near-zero discriminative topical entropy.
5. **Morphological Normalization (Lemmatization):** Uses WordNet Lemmatizer to reduce inflected word variants to their canonical dictionary lemma (e.g., `"shares"`, `"sharing"`, `"shared"` $\rightarrow$ `"share"`), reducing vocabulary dimensionality while preserving semantic context.

### Q3. Why did you choose Lemmatization over Stemming for this project?
**Answer:**  
Stemming (such as the Porter Stemmer) is an empirical heuristic algorithm that trims word suffixes (e.g., `"university"` becomes `"univers"`), frequently producing non-words and causing crude over-stemming or under-stemming. In contrast, Lemmatization utilizes morphological lexical databases (WordNet) and grammatical part-of-speech context to return legitimate vocabulary root forms (e.g., `"better"` $\rightarrow$ `"good"`). While stemming is computationally marginally faster, lemmatization produces cleaner, more interpretable TF-IDF feature tokens without semantic degradation.

### Q4. Explain the mathematics of TF-IDF. What do Term Frequency and Inverse Document Frequency signify?
**Answer:**  
TF-IDF measures the topical relevance of term $t$ within document $d$ relative to a collection $D$:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

- **Term Frequency (TF):** Measures how frequently term $t$ appears in document $d$. To prevent documents that repeat a word 20 times from having 20 times the importance of a document containing it once, we apply **sublinear TF scaling**:
  $$\text{sublinear\_TF}(t, d) = 1 + \log(f_{t, d}) \quad \text{for } f_{t, d} > 0$$
- **Inverse Document Frequency (IDF):** Measures term specificity across the entire corpus of $|D|$ documents:
  $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$$
  Terms appearing in nearly every document receive an IDF approaching 1, while terms concentrated in a single category receive high IDF weights.
- **L2 Normalization:** Each document vector $\mathbf{v}$ is normalized to unit length ($\|\mathbf{v}\|_2 = 1$), eliminating document length bias.

### Q5. What is Data Leakage in text classification, and how did you guarantee its prevention?
**Answer:**  
Data leakage occurs when information from the evaluation/test partition inadvertently contaminates the training phase. In NLP, if `TfidfVectorizer.fit()` is called on the entire corpus before splitting, the vocabulary dictionary, document frequencies ($\text{DF}$), and inverse document frequency weights ($\text{IDF}$) incorporate test set statistics.  
We strictly prevented data leakage by:
1. Splitting the raw text using a stratified 80/20 train-test split (`random_state=42`).
2. Encapsulating `TfidfVectorizer` and each classifier inside an integrated Scikit-learn `Pipeline`.
3. Calling `pipeline.fit()` exclusively on `X_train`. When `pipeline.predict(X_test)` runs, the frozen training vocabulary is used for transformation.

### Q6. Why did you configure `ngram_range=(1, 2)` instead of unigrams alone?
**Answer:**  
Unigrams (individual words) lose all syntactic collocation and contextual sequence. In news articles, multi-word collocations are crucial discriminators:
- *"interest rate"*, *"stock exchange"*, *"wall street"* $\rightarrow$ strongly signify **Business**
- *"prime minister"*, *"downing street"*, *"house common"* $\rightarrow$ strongly signify **Politics**
- *"premier league"*, *"world cup"*, *"champions league"* $\rightarrow$ strongly signify **Sport**  
By including bigrams alongside unigrams, the model captures domain-specific phrasal units without the exponential parameter explosion of higher-order n-grams.

---

## Section B: Four Machine Learning Classifiers

### Q7. Explain how the Decision Tree classifier works for text. Why was it the lowest-performing model in your battle?
**Answer:**  
The Decision Tree (CART) recursively selects the feature token and threshold that maximizes Gini Impurity reduction:

$$I_G(p) = 1 - \sum_{i=1}^C p_i^2$$

In text classification, feature vectors are 5,000-dimensional and highly sparse (over 98% zeros). A Decision Tree can only make orthogonal, axis-aligned splits on one word at a time ($x_j \leq \theta$). Because topical semantics in news documents are diffused across hundreds of correlated terms rather than concentrated in a single pivotal word, the tree fractures into deep, overfitted branches with high variance, resulting in an accuracy of **85.17%** (compared to >97% for the other three algorithms).

### Q8. How does K-Nearest Neighbours (KNN) classify a news article? What is its computational complexity?
**Answer:**  
KNN is a non-parametric, instance-based "lazy learner":
1. During training, it simply stores all 1,780 training feature vectors ($O(1)$ training complexity).
2. During inference, when a new news article arrives, it calculates the distance between the query vector and every single training document, identifies the $k$ closest neighbors ($k=5$), and assigns the majority class weighted by inverse distance ($w = 1/d$).
3. **Computational Complexity:** Inference time is $O(N \cdot d)$, where $N$ is training samples ($1,780$) and $d$ is feature dimensionality ($5,000$). This explains why KNN exhibited the highest inference latency during benchmarking.

### Q9. Why did you use Cosine Distance rather than Euclidean Distance for KNN?
**Answer:**  
In high-dimensional sparse spaces, Euclidean distance suffers severely from the **Curse of Dimensionality**:
$$d_{\text{Euclidean}}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum (u_i - v_i)^2}$$
Because documents vary in length, two articles about the same topic with different word counts will be separated by large Euclidean distances. In contrast, **Cosine Similarity** measures the angle between vectors:

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

For L2-normalized vectors ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1$), Cosine Distance simplifies to $1 - (\mathbf{u} \cdot \mathbf{v})$. It evaluates directional alignment rather than Euclidean magnitude, correctly identifying topical similarity regardless of document length.

### Q10. What is the mathematical formulation of Support Vector Machines (SVM)? Why is it so dominant in text classification?
**Answer:**  
Linear SVM constructs the optimal maximum-margin hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ separating classes by solving the quadratic optimization problem:

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to } y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

SVM dominates text classification because:
1. **High Dimensionality:** Vapnik's Statistical Learning Theory demonstrates that SVM generalization depends on the margin width rather than the number of input dimensions. With 5,000 features, text is almost always linearly separable.
2. **Sparsity:** SVM solutions depend only on the critical **support vectors** lying near class decision boundaries, ignoring irrelevant zero-valued features.
3. **Empirical Result:** SVM achieved the highest 5-Fold Cross Validation score (**98.48%**) and tied for highest test accuracy (**97.98%**).

### Q11. Explain the "Naive" assumption in Multinomial Naive Bayes. If the assumption is false, why does it work so well?
**Answer:**  
Multinomial Naive Bayes applies Bayes' Theorem:

$$P(C_k | \mathbf{x}) \propto P(C_k) \prod_{i=1}^V P(w_i | C_k)^{x_i}$$

The "Naive" assumption is that all word tokens $w_i$ are **conditionally independent given the class label $C_k$**. This assumption is clearly violated in natural language (words like *"interest"* and *"rates"* or *"prime"* and *"minister"* are heavily correlated).  
However, as mathematically proven by Domingos & Pazzani (1997), **accurate classification requires only that the correct class has the highest posterior probability, not that the exact probability estimate is calibrated**. Since correlation shifts probabilities symmetrically across classes, the class ranking remains remarkably accurate, yielding **97.98% test accuracy** and instantaneous training ($0.84$ seconds).

### Q12. What is Laplace Smoothing, and why is it necessary in Naive Bayes?
**Answer:**  
If a word token in the test set never appeared in a specific class during training, the maximum likelihood estimate $P(w_i | C_k)$ becomes 0. Because Naive Bayes multiplies feature probabilities together:

$$\prod_{i=1}^V P(w_i | C_k) = 0$$

A single unseen word would zero-out the entire category probability regardless of all other words! **Laplace/Lidstone Smoothing** adds a pseudo-count $\alpha > 0$ to every numerator:

$$P(w_i | C_k) = \frac{N_{k, i} + \alpha}{N_k + \alpha V}$$

In our pipeline, we tuned $\alpha = 0.1$, which prevents zero-probability bottlenecks while preserving sharp probability peaks.

---

## Section C: Evaluation Metrics & Diagnostic Analysis

### Q13. Define Precision, Recall, and F1-Score in multiclass classification.
**Answer:**  
For any class $c$:
- **Precision:** $\frac{TP_c}{TP_c + FP_c}$ — Proportion of articles assigned to class $c$ that genuinely belong to class $c$ (purity / avoidance of false alarms).
- **Recall:** $\frac{TP_c}{TP_c + FN_c}$ — Proportion of actual class $c$ articles successfully retrieved by the classifier (completeness / sensitivity).
- **F1-Score:** Harmonic mean of Precision and Recall:
  $$\text{F1}_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### Q14. What is the difference between Macro-Averaging and Weighted-Averaging? Which did you prioritize?
**Answer:**  
- **Macro-Averaging:** Calculates the metric independently for each of the 5 classes and takes the unweighted arithmetic mean:
  $$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k$$
  Every category has equal weight, ensuring that small categories are not masked by large ones.
- **Weighted-Averaging:** Weights each class metric by its support (number of true instances in that class):
  $$\text{Weighted-F1} = \frac{1}{N} \sum_{k=1}^K N_k \cdot \text{F1}_k$$
- **Priority:** We prioritized **Macro-F1** (SVM: 0.9796, MNB: 0.9799) because academic benchmarking requires balanced performance across all five categories regardless of class size differences.

### Q15. Interpret the Confusion Matrix of your Support Vector Machine model. Where did the few errors occur?
**Answer:**  
On the 445 held-out test articles, SVM achieved 436 correct predictions (97.98%):
- **Politics:** 84 / 84 correct (**100% Recall**)
- **Sport:** 102 / 102 correct (**100% Recall**)
- **Business:** 98 / 102 correct (3 confused with Politics, 1 with Entertainment)
- **Entertainment:** 76 / 77 correct (1 confused with Sport)
- **Tech:** 76 / 80 correct (2 confused with Business, 1 with Entertainment, 1 with Sport)  
**Error Analysis:** The minor confusion between Business and Politics stemmed from articles discussing government budget deficits and Chancellor Gordon Brown's treasury spending, where financial terms (*"taxes"*, *"economy"*, *"budget"*) overlap heavily with ministerial governance.

### Q16. What was the purpose of 5-Fold Stratified Cross-Validation?
**Answer:**  
A single train-test split can yield optimistic or pessimistic performance estimates based on favorable random sampling. **5-Fold Stratified Cross-Validation** partitions the 1,780 training instances into 5 equal folds preserving exact class distributions. The model trains on 4 folds and validates on the remaining fold across 5 iterations.  
Our cross-validation results confirmed model stability:
- SVM: $98.48\% \pm 0.29\%$
- Naive Bayes: $96.97\% \pm 0.74\%$
- KNN: $96.12\% \pm 1.13\%$
- Decision Tree: $83.54\% \pm 1.59\%$  
SVM exhibited the lowest standard deviation ($\pm 0.29\%$), proving superior generalizability.

---

## Section D: What-If Analysis & Practical Engineering

### Q17. Explain your first What-If experiment: KNN sensitivity to $k$. What happened as $k$ increased?
**Answer:**  
We varied $k \in [1, 3, 5, 7, 9, 11, 15, 21, 31]$.  
- At $k=1$, accuracy was $95.51\%$ because a single nearest neighbor is sensitive to noisy or outlier articles.
- Accuracy peaked at $k=3$ ($97.98\%$) and remained steady around $k=5$ ($97.08\%$).
- Beyond $k=15$, the decision boundary becomes overly smooth, incorporating neighbors from adjacent topical clusters, but distance weighting ($w = 1/d$) cushioned the decline.
- Furthermore, **Cosine metric strictly dominated Euclidean metric**, verifying the necessity of angular similarity in text.

### Q18. Explain your second What-If experiment: TF-IDF `max_features` vocabulary size.
**Answer:**  
We evaluated vocabulary sizes from 250 to Unlimited (60,659 features):
- At 250 features: Accuracy dropped to $94.38\%$ (SVM) and $93.48\%$ (NB) because critical domain-specific vocabulary was truncated.
- At 500 features: Accuracy jumped to $96.85\%$.
- Between 3,000 and 5,000 features: Performance reached an optimal plateau ($97.98\%$).
- Setting `max_features=None` (60,659 features) showed no statistically significant gain ($97.98\%$), while multiplying memory usage and vector dimension twelvefold.  
This empirical defense justifies choosing `max_features=5000`.

### Q19. What is Overfitting, and what explicit mechanisms did you implement to prevent it?
**Answer:**  
Overfitting occurs when an estimator memorizes noise and idiosyncrasies of the training partition at the expense of generalizability on unseen data.  
We prevented overfitting via:
1. **Regularization in SVM:** The $C=1.0$ soft-margin penalty restricts large hyperplane weights $\|\mathbf{w}\|^2$.
2. **Tree Pruning:** We capped `max_depth=50` and enforced `min_samples_split` in the Decision Tree.
3. **Feature Selection:** Ignored rare hapax legomena with `min_df=3` and restricted vocabulary to top 5,000 features.
4. **Sublinear TF Scaling:** Prevented outlier word repetitions from skewing vectors.
5. **Stratified 5-Fold Cross-Validation:** Ensured hyperparameter robustness across folds.

### Q20. If this system were deployed in a real-world high-throughput news agency, which model would you recommend and why?
**Answer:**  
I recommend **Multinomial Naive Bayes** or **Linear SVM**:
- **Inference Speed:** Multinomial Naive Bayes requires only log-probability additions ($0.16$ milliseconds per document), enabling over 6,000 news classifications per second on a single CPU core.
- **Accuracy & Robustness:** Linear SVM delivers the highest Cross-Validation accuracy ($98.48\%$) with $1.18$ ms latency.
- In practice, a two-stage hybrid or an SVM with linear kernel provides the optimal trade-off of commercial accuracy and sub-millisecond throughput.

---
*Prepared for Student Academic Technical Defence – Sathish Kumar R (RA2411026050188)*

