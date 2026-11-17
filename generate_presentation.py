"""
Script to generate the comprehensive, professional academic presentation (.pptx)
Author: Sathish Kumar R (RA2411026050188)
Institution: SRM Institute of Science and Technology, Tiruchirappalli
Course: 21CSE377T - Information Retrieval | CLA-2 Task 2
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PRESENTATION_DIR = os.path.join(PROJECT_ROOT, "presentation")
os.makedirs(PRESENTATION_DIR, exist_ok=True)
PPTX_PATH = os.path.join(PRESENTATION_DIR, "News_Category_Classification_Battle.pptx")

# Color Palette (Deep Academic Navy, Gold, Charcoal, White)
COLOR_NAVY = RGBColor(13, 59, 102)      # #0d3b66
COLOR_DARK_BLUE = RGBColor(0, 30, 61)   # #001e3d
COLOR_GOLD = RGBColor(244, 211, 94)     # #f4d35e
COLOR_CHARCOAL = RGBColor(40, 40, 40)   # #282828
COLOR_LIGHT_GRAY = RGBColor(245, 247, 250)
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_ACCENT = RGBColor(44, 122, 123)


def add_header(slide, title_text, category_text="SRMIST | 21CSE377T – INFORMATION RETRIEVAL"):
    """Adds a standard executive header to a slide."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8.4), Inches(0.9))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0
    
    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_ACCENT
    
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_NAVY


def set_speaker_notes(slide, notes_text):
    """Sets speaker notes for the slide."""
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625) # 16:9 Widescreen standard
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(10), Inches(5.625))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_DARK_BLUE
    bg1.line.color.rgb = COLOR_DARK_BLUE

    # Title box
    tbox = slide1.shapes.add_textbox(Inches(1.0), Inches(1.0), Inches(8.0), Inches(3.6))
    tf = tbox.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY, TIRUCHIRAPPALLI"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_GOLD

    p1 = tf.add_paragraph()
    p1.text = "NEWS CATEGORY CLASSIFICATION\nUSING MACHINE LEARNING"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_before = Pt(14)
    p1.space_after = Pt(14)

    p2 = tf.add_paragraph()
    p2.text = "Text Classification Battle: Compare, Analyze and Defend"
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(200, 220, 240)

    p3 = tf.add_paragraph()
    p3.text = "\nStudent Name: Sathish Kumar R  |  Register No: RA2411026050188\nCourse: 21CSE377T – Information Retrieval  |  CLA-2 Task 2 (30 Marks)"
    p3.font.size = Pt(12)
    p3.font.color.rgb = COLOR_GOLD
    p3.space_before = Pt(18)

    set_speaker_notes(slide1, """Good morning respected evaluator and faculty members. 
My name is Sathish Kumar R, bearing register number RA2411026050188 from SRM Institute of Science and Technology, Tiruchirappalli Campus.
Today, I present my Information Retrieval CLA-2 Task 2 project titled: 'News Category Classification Using Machine Learning: A Comparative Battle Between Four Classifiers'.
In this research defense, I will demonstrate how we ingest the authentic BBC News corpus, preprocess raw text, convert documents into high-dimensional TF-IDF feature spaces without data leakage, and empirically compare Decision Trees, KNN, Support Vector Machines, and Multinomial Naive Bayes.""")

    # =========================================================================
    # SLIDE 2: Problem Statement & Motivation
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "1. Problem Statement & Motivation")
    
    tb2 = slide2.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(8.4), Inches(3.6))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "The Rapid Explosion of Unstructured Text in Information Retrieval:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY

    bullets = [
        ("Information Overload:", "Online news agencies, search engines, and social aggregators publish millions of unlabelled articles daily. Manual indexing is unscalable, subjective, and operationally prohibitive."),
        ("Core IR Challenge:", "Information Retrieval systems require precise topical partitioning (*Business, Entertainment, Politics, Sport, Tech*) to optimize index storage, fast query routing, and personalized recommendations."),
        ("The Classification Battle:", "Different machine learning paradigms make radically distinct mathematical assumptions regarding text representation. Determining which algorithm achieves peak accuracy with minimal latency is vital for production deployments."),
        ("Academic Goal:", "Deliver a transparent, 100% reproducible benchmark adhering strictly to the CLA-2 Task 2 guidelines, proving theoretical claims with empirical evidence.")
    ]
    for title, desc in bullets:
        p = tf2.add_paragraph()
        p.text = f"• {title} {desc}"
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_CHARCOAL
        p.space_before = Pt(8)

    set_speaker_notes(slide2, """In this slide, we examine the problem statement. Automated text categorization is a pillar of modern Information Retrieval. 
Without automated classification, search engines would have to perform brute-force inverted index lookups across the entire web. 
By categorizing documents into topical buckets, we enable federated search, topical routing, and contextual retrieval.
The central question of this task is: when given a standard high-dimensional text corpus, which classifier best navigates feature sparsity, computational latency, and classification accuracy?""")

    # =========================================================================
    # SLIDE 3: Project Objectives
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "2. Project Objectives")
    
    tb3 = slide3.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(8.4), Inches(3.6))
    tf3 = tb3.text_frame
    tf3.word_wrap = True

    objs = [
        "1. Ingest & Audit Dataset: Validate authentic BBC News dataset (2,225 articles, 5 verified categories, zero fabricated records).",
        "2. Multi-Stage Preprocessing: Implement lowercasing, regex symbol stripping, tokenization, NLTK stop-word removal, and WordNet lemmatization.",
        "3. High-Dimensional TF-IDF: Extract unigram and bigram feature representations, applying sublinear scaling and L2 normalization.",
        "4. Strict Leakage Prevention: Ensure TF-IDF vocabulary and IDF weights are fitted strictly on training data using Scikit-Learn Pipelines.",
        "5. Algorithmic Head-to-Head: Train and compare Decision Tree, KNN, Linear SVM, and Multinomial Naive Bayes using stratified 80/20 train-test splits and 5-fold CV.",
        "6. Diagnostic Evaluation: Measure Accuracy, Macro/Weighted Precision, Recall, F1, latency, and confusion matrices.",
        "7. Empirical What-If Analysis: Investigate sensitivity to neighborhood size k, distance metric, vocabulary capacity, and n-grams.",
        "8. Web Dashboard: Deploy an interactive Streamlit application with live explainable news classification."
    ]
    for idx, obj in enumerate(objs):
        p = tf3.paragraphs[0] if idx == 0 else tf3.add_paragraph()
        p.text = obj
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_CHARCOAL
        p.space_before = Pt(6)

    set_speaker_notes(slide3, """These eight objectives define the scope of CLA-2 Task 2. 
Notice that we explicitly focus on methodological rigor: preventing data leakage through integrated pipelines, 
enforcing reproducible random seeds, conducting 5-fold cross-validation, and verifying our findings through empirical What-If sensitivity tests.""")

    # =========================================================================
    # SLIDE 4: BBC News Dataset Provenance & Distribution
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "3. BBC News Dataset Provenance & EDA")

    tb4 = slide4.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(4.5), Inches(3.8))
    tf4 = tb4.text_frame
    tf4.word_wrap = True

    p = tf4.paragraphs[0]
    p.text = "Dataset Specification & Characteristics:"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY

    facts = [
        ("Source:", "D. Greene & P. Cunningham (ICML 2006), University College Dublin benchmark."),
        ("Total Records:", "2,225 genuine news articles (2004–2005)."),
        ("Class Distribution:", "Sport (511), Business (510), Politics (417), Tech (401), Entertainment (386)."),
        ("Missing Values:", "0 null/missing records across entire dataset."),
        ("Duplicate Records:", "98 identical wire updates identified and documented."),
        ("Article Lengths:", "Mean word count ranges from 328 words (Business) to 503 words (Tech).")
    ]
    for k, v in facts:
        p = tf4.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(11)
        p.space_before = Pt(4)

    # Insert class distribution chart if exists
    chart_path = os.path.join(RESULTS_DIR, "class_distribution.png")
    if os.path.exists(chart_path):
        slide4.shapes.add_picture(chart_path, Inches(5.4), Inches(1.5), width=Inches(4.0))

    set_speaker_notes(slide4, """Slide 4 presents our dataset exploration. We use the genuine BBC News corpus published by Greene and Cunningham at ICML 2006. 
The dataset contains 2,225 articles distributed across 5 categories. 
Notice that the classes are well-balanced, ranging between 17.4% and 23.0%. 
Our data audit confirmed zero missing values and uncovered 98 syndicated news duplicates. 
Tech and Politics articles are noticeably longer than Business and Sport articles, averaging approximately 500 words per story.""")

    # =========================================================================
    # SLIDE 5: Text Preprocessing Pipeline
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "4. Multi-Stage Linguistic Preprocessing")

    tb5 = slide5.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(3.8))
    tf5 = tb5.text_frame
    tf5.word_wrap = True

    p = tf5.paragraphs[0]
    p.text = "Sequential Preprocessing Operations:"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY

    steps = [
        ("1. Lowercase Normalization:", "Maps all characters to lowercase (eliminates 'Apple' vs 'apple' divergence)."),
        ("2. Regex Token Sanitization:", "Strips URLs, HTML tags, punctuation, and numerical digits, isolating alphabetic text."),
        ("3. Word Tokenization:", "Splits continuous sentence strings into lexical tokens."),
        ("4. Stop-Word Removal:", "Prunes high-frequency English grammatical particles (NLTK stop-word lexicon)."),
        ("5. WordNet Lemmatization:", "Reduces inflected tokens to canonical dictionary lemmas ('shares' -> 'share')."),
        ("Dimensionality Reduction:", "Achieves an average 35% to 50% token reduction per document, stripping noise while preserving topical semantics.")
    ]
    for k, v in steps:
        p = tf5.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(11)
        p.space_before = Pt(4)

    p_demo = tf5.add_paragraph()
    p_demo.text = "\nExample Transformation:"
    p_demo.font.size = Pt(11.5)
    p_demo.font.bold = True
    p_demo.font.color.rgb = COLOR_NAVY

    p_ex1 = tf5.add_paragraph()
    p_ex1.text = "RAW: 'The UK Chancellor Gordon Brown announced a £1.5 billion investment in artificial intelligence!'"
    p_ex1.font.size = Pt(10)
    p_ex1.font.color.rgb = RGBColor(180, 40, 40)

    p_ex2 = tf5.add_paragraph()
    p_ex2.text = "CLEAN: 'chancellor gordon brown announced billion investment artificial intelligence'"
    p_ex2.font.size = Pt(10)
    p_ex2.font.bold = True
    p_ex2.font.color.rgb = RGBColor(20, 120, 40)

    set_speaker_notes(slide5, """Here we review our text preprocessing pipeline. 
Raw text is noisy, containing HTML entities, British currency symbols, capitalization, and punctuation. 
Our modular TextPreprocessor applies five rigorous stages. 
Notice the before-and-after example at the bottom: 
WordNet lemmatization maps words to their true lexical roots, and stop-word filtering strips grammatical glue words. 
This reduces input dimensionality by over 35% while preserving discriminative semantic tokens.""")

    # =========================================================================
    # SLIDE 6: TF-IDF Feature Representation & Leakage Prevention
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "5. TF-IDF Mathematical Representation")

    tb6 = slide6.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(3.8))
    tf6 = tb6.text_frame
    tf6.word_wrap = True

    formulas = [
        ("Term Frequency with Sublinear Scaling:", "sublinear_TF(t, d) = 1 + log(f_t,d) for f_t,d > 0. Dampens the influence of repeatedly occurring terms."),
        ("Inverse Document Frequency:", "IDF(t, D) = log((1 + |D|) / (1 + DF(t))) + 1. Rewards discriminative topical words while penalizing universally ubiquitous words."),
        ("Composite L2 Normalization:", "v(d) = w(d) / ||w(d)||_2. Normalizes document vectors to unit length on hypersphere, neutralizing document length discrepancies."),
        ("N-Gram Range (1, 2):", "Captures both unigrams ('inflation') and essential bigrams ('interest rate', 'prime minister', 'mobile phone')."),
        ("STRICT DATA LEAKAGE PREVENTION:", "TF-IDF vocabulary and IDF weights are computed EXCLUSIVELY on X_train inside a Scikit-Learn Pipeline. X_test is transformed using frozen training parameters, guaranteeing zero test set leakage.")
    ]
    for idx, (k, v) in enumerate(formulas):
        p = tf6.paragraphs[0] if idx == 0 else tf6.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_CHARCOAL
        p.space_before = Pt(6)

    set_speaker_notes(slide6, """Slide 6 presents the mathematical core of Information Retrieval: TF-IDF vectorization. 
We map text into a 5,000-dimensional sparse space. 
We use sublinear TF scaling, which logarithmically dampens raw frequencies so that an article mentioning 'shares' five times does not receive five times the weight of an article mentioning it once. 
Crucially, look at the bottom point: we prevent data leakage by enclosing TfidfVectorizer in an integrated Pipeline. 
The test set is completely unseen until inference time.""")

    # =========================================================================
    # SLIDE 7: Four Competing Classifiers
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "6. The Four Competing Classifiers")

    tb7 = slide7.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(3.8))
    tf7 = tb7.text_frame
    tf7.word_wrap = True

    models_info = [
        ("1. Decision Tree (CART):", "Non-linear tree partitioning based on Gini impurity. Splitting criterion: I_G = 1 - sum(p_i^2). Max depth = 50. Makes orthogonal axis-aligned splits on individual words."),
        ("2. K-Nearest Neighbours (KNN):", "Instance-based non-parametric lazy learner. Computes Cosine similarity against all training instances; assigns majority class among k=5 neighbors weighted by inverse distance (w = 1/d)."),
        ("3. Support Vector Machine (Linear SVM):", "Finds the maximum-margin hyperplane w^T x + b = 0 minimizing 1/2||w||^2 + C * sum(xi_i). Optimal for high-dimensional, linearly separable sparse feature spaces."),
        ("4. Multinomial Naive Bayes (MNB):", "Probabilistic classifier applying Bayes' theorem: P(c|d) proportional to P(c) * prod(P(w|c)). Uses Laplace smoothing (alpha=0.1) to eliminate zero-frequency errors.")
    ]
    for idx, (k, v) in enumerate(models_info):
        p = tf7.paragraphs[0] if idx == 0 else tf7.add_paragraph()
        p.text = f"{k} {v}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_CHARCOAL
        p.space_before = Pt(8)

    set_speaker_notes(slide7, """In Slide 7, we outline the theoretical foundation of the four competing algorithms. 
Decision Trees construct hierarchical decision rules on isolated words. 
KNN compares query vectors against all training documents using cosine similarity. 
Linear SVM finds the maximum-margin hyperplane separating classes in 5,000-dimensional space. 
Multinomial Naive Bayes estimates posterior class likelihoods using word frequency counts and Laplace smoothing. 
Notice the rich diversity in theoretical assumptions across these four models.""")

    # =========================================================================
    # SLIDE 8: Experimental Setup & Cross-Validation
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "7. Experimental Setup & Protocol")

    tb8 = slide8.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(3.8))
    tf8 = tb8.text_frame
    tf8.word_wrap = True

    protocols = [
        ("Stratified Split Ratio:", "80% Training (1,780 articles) vs 20% Testing (445 articles). Stratified sampling guarantees identical class proportions across both splits."),
        ("Reproducibility:", "Fixed random seed random_state=42 enforced across all splits and stochastic estimators."),
        ("Cross-Validation Protocol:", "5-Fold Stratified Cross-Validation executed exclusively on the training partition to estimate generalization variance and model stability."),
        ("Evaluation Metric Suite:", "Accuracy, Macro Precision, Macro Recall, Macro F1-Score, Weighted F1-Score, Training Run Time, Inference Query Latency (ms/sample)."),
        ("Fairness Principle:", "All four classifiers utilize the exact same preprocessed text, identical TF-IDF feature vocabulary, and identical stratified test partition.")
    ]
    for idx, (k, v) in enumerate(protocols):
        p = tf8.paragraphs[0] if idx == 0 else tf8.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_CHARCOAL
        p.space_before = Pt(6)

    set_speaker_notes(slide8, """Slide 8 details our rigorous experimental setup. 
To ensure a completely fair battle, every classifier was evaluated on the exact same 445 test articles with identical preprocessing. 
We performed 5-Fold Stratified Cross-Validation on the training set. 
Notice that we evaluate both Macro and Weighted metrics, and we benchmark computational run-times to assess practical deployment feasibility.""")

    # =========================================================================
    # SLIDE 9: Official Experimental Leaderboard (Actual Results)
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "8. Experimental Results: Head-to-Head Battle")

    # Insert Leaderboard Table
    rows, cols = 5, 7
    left, top, width, height = Inches(0.8), Inches(1.5), Inches(8.4), Inches(2.2)
    table_shape = slide9.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    # Column widths
    table.columns[0].width = Inches(2.2)
    for c in range(1, cols):
        table.columns[c].width = Inches(1.03)

    headers = ["Classifier", "Accuracy", "Macro Prec", "Macro Rec", "Macro F1", "5-Fold CV", "Latency"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = COLOR_WHITE
            p.alignment = PP_ALIGN.CENTER

    data = [
        ["Multinomial Naive Bayes", "97.98%", "0.9791", "0.9813", "0.9799", "96.97%", "0.16 ms"],
        ["Support Vector Machine", "97.98%", "0.9801", "0.9796", "0.9796", "98.48%", "1.18 ms"],
        ["K-Nearest Neighbours", "97.08%", "0.9708", "0.9712", "0.9708", "96.12%", "0.21 ms"],
        ["Decision Tree", "85.17%", "0.8534", "0.8521", "0.8524", "83.54%", "0.16 ms"]
    ]
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_WHITE if r % 2 == 0 else COLOR_LIGHT_GRAY
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(9.5)
                p.font.color.rgb = COLOR_CHARCOAL
                if c == 0:
                    p.alignment = PP_ALIGN.LEFT
                    p.font.bold = True
                else:
                    p.alignment = PP_ALIGN.CENTER

    # Summary callout below table
    tb9 = slide9.shapes.add_textbox(Inches(0.8), Inches(4.0), Inches(8.4), Inches(1.2))
    tf9 = tb9.text_frame
    tf9.word_wrap = True
    p = tf9.paragraphs[0]
    p.text = "Key Empirical Findings:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY

    p2 = tf9.add_paragraph()
    p2.text = "• SVM & Naive Bayes tied for top test accuracy at 97.98% (436/445 articles correct).\n• SVM demonstrated highest 5-Fold Cross Validation stability (98.48% +/- 0.29%).\n• Naive Bayes delivered ultra-low latency (0.16 ms/doc), over 7x faster inference than calibrated SVM."
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = COLOR_CHARCOAL

    set_speaker_notes(slide9, """Slide 9 displays our actual experimental results. 
Multinomial Naive Bayes and Support Vector Machine tied for the championship with 97.98% test accuracy, 
correctly classifying 436 out of 445 test documents. 
Looking across all metrics, SVM achieved the highest 5-fold cross-validation accuracy of 98.48% with the lowest variance. 
Naive Bayes demonstrated exceptional computational throughput, running in 0.16 ms per document. 
Decision Tree lagged significantly at 85.17% due to feature fragmentation in sparse high-dimensional space.""")

    # =========================================================================
    # SLIDE 10: Comparative Visualizations & Speed Trade-off
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "9. Comparative Performance Visualizations")

    c1 = os.path.join(RESULTS_DIR, "model_comparison_bar.png")
    c2 = os.path.join(RESULTS_DIR, "training_prediction_time.png")

    if os.path.exists(c1):
        slide10.shapes.add_picture(c1, Inches(0.8), Inches(1.5), width=Inches(4.2))
    if os.path.exists(c2):
        slide10.shapes.add_picture(c2, Inches(5.1), Inches(1.5), width=Inches(4.2))

    set_speaker_notes(slide10, """In Slide 10, we examine our generated comparative charts. 
On the left, the bar chart illustrates how SVM, Naive Bayes, and KNN maintain near-perfect consistency across Accuracy, Macro Precision, Recall, and F1-score, while Decision Tree drops noticeably. 
On the right, the computational latency chart reveals that SVM takes 12.5 seconds to train due to probability calibration, whereas Naive Bayes trains in 0.84 seconds and executes queries in sub-millisecond time.""")

    # =========================================================================
    # SLIDE 11: Confusion Matrix Diagnostics & Error Analysis
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "10. Confusion Matrix & Diagnostic Error Analysis")

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrices_all.png")
    if os.path.exists(cm_path):
        slide11.shapes.add_picture(cm_path, Inches(0.8), Inches(1.4), width=Inches(4.5))

    tb11 = slide11.shapes.add_textbox(Inches(5.4), Inches(1.4), Inches(4.0), Inches(3.8))
    tf11 = tb11.text_frame
    tf11.word_wrap = True

    p = tf11.paragraphs[0]
    p.text = "Per-Class Error Diagnostics:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY

    diag = [
        ("100% Recall in Sport & Politics:", "SVM achieved perfect 102/102 in Sport and 84/84 in Politics due to highly distinct lexical signatures ('arsenal', 'blair', 'cup')."),
        ("Minor Business/Politics Overlap:", "4 Business articles were misclassified as Politics because treasury budgets, corporate tax legislation, and government fiscal policy span both domains."),
        ("Tech/Business Overlap:", "2 Tech articles discussing corporate revenue of telecommunication giants were classified under Business."),
        ("Decision Tree Degradation:", "Decision Tree exhibited cross-category errors in Business (81/102) and Entertainment (66/77).")
    ]
    for k, v in diag:
        p = tf11.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(10)
        p.space_before = Pt(4)

    set_speaker_notes(slide11, """Slide 11 breaks down the confusion matrices. 
For our champion SVM model, out of 445 test articles, only 9 errors were made. 
Sport and Politics achieved 100% recall. 
The four misclassifications between Business and Politics are linguistically understandable: 
articles discussing Gordon Brown's UK budget deficit and economic taxation policies naturally bridge both finance and governance.""")

    # =========================================================================
    # SLIDE 12: Empirical What-If Parametric Analysis
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    add_header(slide12, "11. Empirical What-If Sensitivity Analysis")

    whatif_img = os.path.join(RESULTS_DIR, "what_if_analysis.png")
    if os.path.exists(whatif_img):
        slide12.shapes.add_picture(whatif_img, Inches(0.8), Inches(1.4), width=Inches(5.0))

    tb12 = slide12.shapes.add_textbox(Inches(5.9), Inches(1.4), Inches(3.6), Inches(3.8))
    tf12 = tb12.text_frame
    tf12.word_wrap = True

    p = tf12.paragraphs[0]
    p.text = "Experimental Insights:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY

    w_insights = [
        ("Exp 1 (KNN k & Distance):", "Cosine distance strictly outperforms Euclidean distance across all k. Accuracy peaks at k=3 (97.98%) and stabilizes at k=5 (97.08%)."),
        ("Exp 2 (Vocabulary Size):", "Accuracy surges rapidly from 250 to 3,000 features. Beyond 5,000 features, performance plateaus. Setting max_features=None (60,659 terms) yields no gain while increasing memory 12x."),
        ("Exp 3 (N-Gram Lexical Span):", "Bigrams (1,2) provide noticeable accuracy gains over unigrams (1,1) by preserving phrasal collocations ('prime minister', 'wall street').")
    ]
    for k, v in w_insights:
        p = tf12.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(9.5)
        p.space_before = Pt(4)

    set_speaker_notes(slide12, """Slide 12 presents our real What-If sensitivity experiments. 
First, we varied KNN neighbors from 1 to 31: Cosine distance proved strictly superior to Euclidean distance because it evaluates angular orientation rather than document length. 
Second, testing TF-IDF vocabulary capacities from 250 to 60,000 terms proved that performance plateaus at 5,000 features. 
Third, adding bigrams boosted classification by capturing essential two-word phrases.""")

    # =========================================================================
    # SLIDE 13: Streamlit Interactive Dashboard
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    add_header(slide13, "12. Streamlit Interactive Web Application")

    tb13 = slide13.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(3.8))
    tf13 = tb13.text_frame
    tf13.word_wrap = True

    p = tf13.paragraphs[0]
    p.text = "Interactive Application Architecture (app.py):"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY

    features = [
        ("1. Home & Executive Overview:", "Student identification, project motivation, system architecture flowchart."),
        ("2. Dataset Exploration:", "Interactive class distributions, boxplots, word count statistics, raw article browser."),
        ("3. Preprocessing Lab:", "Step-by-step transformation viewer with live token reduction calculations."),
        ("4. TF-IDF Feature Space:", "Mathematical formulas, L2 normalization breakdown, top category keywords."),
        ("5. Four-Model Battle Arena:", "Leaderboard tables, comparative bar charts, and performance radar plots."),
        ("6. Confusion Matrix Diagnostics:", "Interactive confusion matrix selector with normalized percentages."),
        ("7. What-If Parametric Explorer:", "Interactive charts for KNN k, max_features, and n-grams."),
        ("8. Live News Classifier:", "Real-time inference engine with posterior probability distribution bars and influential keyword extraction.")
    ]
    for k, v in features:
        p = tf13.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_CHARCOAL
        p.space_before = Pt(3)

    set_speaker_notes(slide13, """Slide 13 showcases our Streamlit web application. 
We built an 8-page interactive dashboard. 
Users can explore the dataset, observe the preprocessing pipeline step-by-step, review the model battle arena, 
and input custom breaking news text to receive instantaneous predictions, probability confidence scores, 
and the exact driving TF-IDF keywords that influenced the classification.""")

    # =========================================================================
    # SLIDE 14: Conclusion, Limitations & Future Scope
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    add_header(slide14, "13. Conclusion, Limitations & Future Scope")

    tb14 = slide14.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(3.8))
    tf14 = tb14.text_frame
    tf14.word_wrap = True

    p = tf14.paragraphs[0]
    p.text = "Conclusions & Technical Defense Summary:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY

    c_points = [
        ("Algorithmic Champion:", "Linear SVM achieved the most robust 5-Fold Cross Validation performance (98.48%) while tying with Naive Bayes at 97.98% test accuracy."),
        ("High-Throughput Production Recommendation:", "Multinomial Naive Bayes is recommended for commercial news aggregators processing thousands of documents per second due to its 0.16 ms latency."),
        ("Project Limitations:", "The dataset reflects UK news context from 2004–2005; vocabulary does not encompass contemporary colloquialisms or modern social media abbreviations."),
        ("Future Scope:", "Explore Dense Transformer Embeddings (BERT, RoBERTa), fine-tune modern LLMs, and implement active learning for continuous human-in-the-loop streaming data.")
    ]
    for k, v in c_points:
        p = tf14.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(10.5)
        p.space_before = Pt(4)

    set_speaker_notes(slide14, """In conclusion, our text classification battle demonstrated that Linear SVM and Multinomial Naive Bayes are the superior architectures for high-dimensional TF-IDF news categorization. 
SVM offers the highest theoretical margin stability, while Naive Bayes offers unparalleled inference throughput. 
Future work can extend this architecture to dense transformer embeddings like BERT for cross-lingual and social media news classification.""")

    # =========================================================================
    # SLIDE 15: Academic References & Defense
    # =========================================================================
    slide15 = prs.slides.add_slide(blank_layout)
    add_header(slide15, "14. Academic References & Technical Defense")

    tb15 = slide15.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(3.8))
    tf15 = tb15.text_frame
    tf15.word_wrap = True

    refs = [
        "1. Greene, D., & Cunningham, P. (2006). Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering. In Proc. 23rd International Conference on Machine Learning (ICML 2006), pp. 377-384.",
        "2. Joachims, T. (1998). Text Categorization with Support Vector Machines: Learning with Many Relevant Features. In European Conference on Machine Learning (ECML), Springer, pp. 137-142.",
        "3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.",
        "4. Domingos, P., & Pazzani, M. (1997). On the Optimality of the Simple Bayesian Classifier under Zero-One Loss. Machine Learning, 29(2), pp. 103-130.",
        "5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, pp. 2825-2830.",
        "6. Salton, G., & Buckley, C. (1988). Term-Weighting Approaches in Automatic Text Retrieval. Information Processing & Management, 24(5), pp. 513-523."
    ]
    for idx, ref in enumerate(refs):
        p = tf15.paragraphs[0] if idx == 0 else tf15.add_paragraph()
        p.text = ref
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_CHARCOAL
        p.space_before = Pt(5)

    p_end = tf15.add_paragraph()
    p_end.text = "\nThank you! Ready for Technical Defence & Viva-Voce Questions."
    p_end.font.size = Pt(13)
    p_end.font.bold = True
    p_end.font.color.rgb = COLOR_NAVY
    p_end.space_before = Pt(10)

    set_speaker_notes(slide15, """These are the key academic foundational references supporting this work, including Greene and Cunningham's original ICML paper introducing the BBC dataset, Thorsten Joachims' landmark paper on SVMs for text, and Manning's Information Retrieval textbook. 
I am now prepared to answer any technical questions regarding my implementation, mathematics, and experimental results. Thank you.""")

    prs.save(PPTX_PATH)
    print(f"Presentation saved successfully to: {PPTX_PATH}")


if __name__ == "__main__":
    create_presentation()

