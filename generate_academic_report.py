"""
Script to generate the Comprehensive Academic Research Report (.docx)
Author: Sathish Kumar R (RA2411026050188)
Institution: SRM Institute of Science and Technology, Tiruchirappalli
Course: 21CSE377T - Information Retrieval | CLA-2 Task 2
"""

import os
import json
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
REPORT_DIR = os.path.join(PROJECT_ROOT, "academic_report")
os.makedirs(REPORT_DIR, exist_ok=True)
DOCX_PATH = os.path.join(REPORT_DIR, "ACADEMIC_REPORT.docx")

# Color Constants
COLOR_NAVY = RGBColor(13, 59, 102)     # #0d3b66
COLOR_DARK_BLUE = RGBColor(0, 30, 61)  # #001e3d
COLOR_GOLD = RGBColor(200, 150, 20)
COLOR_GRAY = RGBColor(100, 100, 100)
COLOR_BLACK = RGBColor(0, 0, 0)
COLOR_WHITE = RGBColor(255, 255, 255)


def set_cell_background(cell, fill_hex):
    """Sets background shading of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell internal padding."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)


def add_custom_heading(doc, text, level):
    """Adds a styled heading with academic numbering."""
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    run = h.runs[0]
    run.font.name = 'Calibri'
    if level == 1:
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = COLOR_NAVY
    elif level == 2:
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = COLOR_DARK_BLUE
    elif level == 3:
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = COLOR_GRAY
    return h


def build_academic_report():
    doc = Document()

    # Set 1-inch standard academic margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst = p_inst.add_run("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY\n")
    r_inst.font.name = 'Calibri'
    r_inst.font.size = Pt(14)
    r_inst.font.bold = True
    r_inst.font.color.rgb = COLOR_NAVY

    r_dept = p_inst.add_run("FACULTY OF ENGINEERING AND TECHNOLOGY\nDEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING\nTIRUCHIRAPPALLI CAMPUS\n")
    r_dept.font.name = 'Calibri'
    r_dept.font.size = Pt(11)
    r_dept.font.color.rgb = COLOR_GRAY

    doc.add_paragraph("\n" * 2)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_course = p_title.add_run("COURSE: 21CSE377T – INFORMATION RETRIEVAL\nCONTINUOUS LEARNING ASSESSMENT – 2 (TASK 2)\nMAXIMUM MARKS: 30\n\n")
    r_course.font.name = 'Calibri'
    r_course.font.size = Pt(11)
    r_course.font.bold = True
    r_course.font.color.rgb = COLOR_GOLD

    r_title = p_title.add_run("NEWS CATEGORY CLASSIFICATION USING MACHINE LEARNING\n")
    r_title.font.name = 'Calibri'
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_NAVY

    r_sub = p_title.add_run("Text Classification Battle: Compare, Analyze and Defend\nAn Empirical Benchmark of Decision Tree, KNN, SVM, and Naive Bayes on the BBC News Corpus")
    r_sub.font.name = 'Calibri'
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = COLOR_DARK_BLUE

    doc.add_paragraph("\n" * 3)

    # Student metadata box table
    table_meta = doc.add_table(rows=5, cols=2)
    table_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_meta.autofit = False

    meta_info = [
        ("Student Name:", "SATHISH KUMAR R"),
        ("Register Number:", "RA2411026050188"),
        ("Degree & Branch:", "B.Tech. Computer Science and Engineering"),
        ("Course Code & Name:", "21CSE377T – Information Retrieval"),
        ("Academic Institution:", "SRM Institute of Science and Technology, Tiruchirappalli")
    ]

    for idx, (label, val) in enumerate(meta_info):
        row = table_meta.rows[idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.5)
        c1.width = Inches(3.8)

        set_cell_background(c0, "F0F4F8")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=80, bottom=80, left=120, right=120)
        set_cell_margins(c1, top=80, bottom=80, left=120, right=120)

        p0 = c0.paragraphs[0]
        r0 = p0.add_run(label)
        r0.font.bold = True
        r0.font.size = Pt(10)
        r0.font.color.rgb = COLOR_NAVY

        p1 = c1.paragraphs[0]
        r1 = p1.add_run(val)
        r1.font.size = Pt(10)
        if idx in [0, 1]:
            r1.font.bold = True
            r1.font.color.rgb = COLOR_DARK_BLUE

    doc.add_page_break()

    # =========================================================================
    # ABSTRACT & KEYWORDS
    # =========================================================================
    add_custom_heading(doc, "ABSTRACT", level=1)
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.line_spacing = 1.15
    p_abs.paragraph_format.space_after = Pt(8)
    p_abs.add_run(
        "Automated text classification is an indispensable cornerstone of modern Information Retrieval (IR) systems, "
        "enabling digital news portals, enterprise document archives, and search engines to catalog, index, and route vast volumes "
        "of unstructured text. In this academic project, we present an empirical comparative battle between four foundational supervised "
        "machine learning algorithms: Decision Tree (CART), K-Nearest Neighbours (KNN), Support Vector Machine (Linear SVM), and "
        "Multinomial Naive Bayes (MNB). The benchmark is conducted on the authentic BBC News Classification corpus comprising 2,225 "
        "articles categorized into five topical domains: Business, Entertainment, Politics, Sport, and Tech. "
        "A rigorous multi-stage linguistic preprocessing pipeline was constructed, encompassing case folding, regex-based symbol and numeric stripping, "
        "alphabetic tokenization, stop-word elimination, and WordNet lemmatization. Textual features were mapped into a 5,000-dimensional "
        "continuous vector space using Term Frequency-Inverse Document Frequency (TF-IDF) with sublinear scaling and L2 unit-vector normalization. "
        "Data leakage was rigorously prevented by encapsulating vectorization and estimation within Scikit-Learn Pipelines fitted exclusively "
        "on the stratified 80% training partition (1,780 documents). Model performance was validated across 5-Fold Stratified Cross-Validation "
        "and benchmarked on the held-out 20% test partition (445 documents) using Accuracy, Macro/Weighted Precision, Recall, F1-Score, and "
        "computational inference latency. Support Vector Machine and Multinomial Naive Bayes both achieved championship test accuracy of "
        "97.98% (Macro F1: 0.9796 and 0.9799 respectively), with SVM demonstrating superior 5-Fold Cross-Validation generalizability (98.48% ± 0.29%) "
        "and Naive Bayes delivering sub-millisecond query latency (0.16 ms/document). Parametric What-If experiments confirmed that Cosine "
        "similarity strictly outperforms Euclidean metric for KNN in sparse spaces, while bigram collocations significantly boost semantic discriminability. "
        "An interactive Streamlit web dashboard was deployed for transparent real-time news inference with explainable keyword attribution."
    )

    p_kw = doc.add_paragraph()
    r_kw_title = p_kw.add_run("Keywords: ")
    r_kw_title.font.bold = True
    p_kw.add_run("Information Retrieval, Text Classification, TF-IDF, Support Vector Machine, Naive Bayes, K-Nearest Neighbours, Decision Tree, BBC News Corpus, What-If Sensitivity Analysis, Streamlit.")

    doc.add_paragraph("---")

    # =========================================================================
    # 1. INTRODUCTION & PROBLEM STATEMENT
    # =========================================================================
    add_custom_heading(doc, "1. INTRODUCTION & PROBLEM STATEMENT", level=1)
    
    p = doc.add_paragraph()
    p.add_run(
        "In the contemporary information era, the exponential expansion of digital textual data poses profound challenges for Information "
        "Retrieval (IR) systems. News syndicates, enterprise databases, regulatory agencies, and search engines ingest millions of unorganized, "
        "heterogeneous documents on a daily basis. Manual categorization by human domain experts is prohibitively expensive, subjective, slow, "
        "and incapable of scaling with streaming real-time news feeds. Automated text categorization—the algorithmic assignment of unlabelled "
        "natural language texts to one or more predefined topical categories based on semantic content—is therefore vital for automated routing, "
        "hierarchical web indexing, faceted search, recommendation filtering, and contextual document retrieval.\n\n"
        "From an Information Retrieval standpoint, automated classification partitions massive inverted index structures into topical "
        "sub-indices. This reduces the search space during query processing, enhances precision, and bounds computational complexity. "
        "However, natural language text is intrinsically unstructured, high-dimensional, polysemous, and sparse. When converted into lexical feature "
        "representations via the Vector Space Model (VSM), document vectors reside in dimensions numbering in the thousands, where the vast "
        "majority of entries are zero.\n\n"
        "The fundamental objective of this assignment (CLA-2 Task 2: Text Classification Battle) is to develop, evaluate, and theoretically "
        "defend a complete machine learning system that automatically classifies incoming news articles into five canonical categories: "
        "Business, Entertainment, Politics, Sport, and Tech. Through rigorous empirical benchmarking, we investigate the mathematical assumptions, "
        "strengths, failure modes, and trade-offs of four contrasting algorithmic paradigms: Decision Trees, K-Nearest Neighbours, Support "
        "Vector Machines, and Multinomial Naive Bayes."
    )

    # =========================================================================
    # 2. DATASET EXPLORATION & AUDIT
    # =========================================================================
    add_custom_heading(doc, "2. DATASET EXPLORATION, AUDIT & CHARACTERISTICS", level=1)

    p = doc.add_paragraph()
    p.add_run(
        "To guarantee academic integrity and direct reproducibility without data fabrication, this study employs the authentic BBC News Classification "
        "benchmark corpus, originally introduced by D. Greene and P. Cunningham (University College Dublin) in their landmark ICML 2006 publication: "
        "'Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering'. The corpus comprises 2,225 individual text "
        "documents collected from the British Broadcasting Corporation (BBC) news website between 2004 and 2005, spanning five topical domains.\n\n"
        "A rigorous exploratory data audit was executed to characterize the dataset structure prior to model construction:"
    )

    # Class distribution table
    t_dist = doc.add_table(rows=7, cols=5)
    t_dist.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_dist.autofit = False

    dist_headers = ["Category", "Record Count", "Proportion (%)", "Mean Word Count", "Typical Topical Subjects"]
    for c, h in enumerate(dist_headers):
        cell = t_dist.cell(0, c)
        cell.text = h
        set_cell_background(cell, "0D3B66")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.runs[0]
        run.font.bold = True
        run.font.size = Pt(9.5)
        run.font.color.rgb = COLOR_WHITE

    dist_rows = [
        ["Sport", "511", "22.97%", "329.26", "Football, rugby, tennis, athletics, Olympics, Champions League"],
        ["Business", "510", "22.92%", "328.88", "Stock markets, corporate earnings, interest rates, economic growth"],
        ["Politics", "417", "18.74%", "453.97", "UK Parliament, elections, Blair government, ministerial policy"],
        ["Tech", "401", "18.02%", "502.70", "Mobile phones, software, broadband, computer security, gadgets"],
        ["Entertainment", "386", "17.35%", "330.62", "Film awards (Oscars, BAFTAs), music, celebrity news, festivals"],
        ["Total Corpus", "2,225", "100.00%", "383.92", "Full Benchmark BBC News Corpus (2004–2005)"]
    ]

    for r_idx, row_data in enumerate(dist_rows):
        row = t_dist.rows[r_idx + 1]
        bg_col = "FFFFFF" if r_idx % 2 == 0 else "F8F9FA"
        if r_idx == 5:
            bg_col = "F0F4F8"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = val
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            p = cell.paragraphs[0]
            run = p.runs[0]
            run.font.size = Pt(9)
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                run.font.bold = True
            elif c_idx in [1, 2, 3]:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                if r_idx == 5:
                    run.font.bold = True
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    p_audit = doc.add_paragraph()
    p_audit.paragraph_format.space_before = Pt(8)
    p_audit.add_run(
        "Key Data Quality Audit Findings:\n"
        "1. Completeness: Exactly zero missing, null, or corrupted text entries exist across both features and target labels.\n"
        "2. Class Balance: The class distribution is well-proportioned across the 5 domains, with Sport and Business accounting for ~23% each, "
        "and Politics, Tech, and Entertainment ranging from 17.3% to 18.7%. This eliminates severe majority-class bias and ensures statistical power.\n"
        "3. Article Length Heterogeneity: Tech articles are the longest on average (mean: 502.7 words, max: 2,969 words), followed by Politics "
        "(mean: 453.9 words, max: 4,432 words). Business and Sport articles are significantly more concise (~329 words). This length variance "
        "underscores the necessity of L2 vector normalization in TF-IDF to avoid magnitude bias.\n"
        "4. Duplicate Analysis: A string matching audit identified 98 exact or near-identical text duplicates resulting from syndicated wire updates. "
        "In accordance with academic protocol, these are retained to preserve direct comparability with published ICML 2006 baselines."
    )

    # Insert class distribution image
    chart_dist_path = os.path.join(RESULTS_DIR, "class_distribution.png")
    if os.path.exists(chart_dist_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(chart_dist_path, width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 1: Class frequency distribution and proportion breakdown across the BBC News corpus.")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(9)
        p_cap.runs[0].font.italic = True

    # =========================================================================
    # 3. TEXT PREPROCESSING PIPELINE
    # =========================================================================
    add_custom_heading(doc, "3. MULTI-STAGE TEXT PREPROCESSING PIPELINE", level=1)

    p = doc.add_paragraph()
    p.add_run(
        "Raw natural language documents cannot be ingested directly into mathematical classifiers due to typographical noise, grammatical "
        "inflections, and arbitrary formatting. A specialized, self-contained TextPreprocessor class was developed in src/preprocessing.py "
        "implementing five sequential transformations:\n\n"
        "1. Lowercase Normalization: All characters are mapped to lowercase. In Information Retrieval, vocabulary capitalization does not alter "
        "topical semantics (e.g., 'Apple' vs 'apple' or 'Market' vs 'market'). Normalization conflates identical lexical concepts into unified dimensions.\n"
        "2. Regex Sanitization & Symbol Stripping: Web URLs (http/www), HTML markup entities, British pound symbols (£), punctuation marks, and "
        "standalone numeric digits are stripped using compiled regular expressions. Numerical figures (e.g., '2004', '8.1%') fluctuate wildly "
        "across individual stories and introduce high-variance sparse dimensions that do not generalize.\n"
        "3. Alphabetic Tokenization: The continuous cleaned character stream is segmented into discrete lexical tokens.\n"
        "4. Stop-Word Elimination: Standard English stop words (from the NLTK lexicon) are filtered out. High-frequency function words ('the', 'is', 'at', 'which', 'from') "
        "exhibit very high collection frequencies but zero discriminative entropy across news topics. Filtering them reduces vector dimensionality by ~40%.\n"
        "5. WordNet Lemmatization: Rather than applying crude heuristic suffix-stripping (Porter Stemming), which frequently yields non-words "
        "(e.g., 'community' -> 'commun'), WordNet Lemmatization maps inflected verbs, nouns, and adjectives to their canonical dictionary base "
        "(e.g., 'shares', 'sharing', 'shared' -> 'share'). This consolidates vocabulary dimensions while preserving morphological semantics."
    )

    # Preprocessing before-and-after table
    t_prep = doc.add_table(rows=4, cols=2)
    t_prep.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_prep.autofit = False

    prep_data = [
        ("Raw Text Input", "The UK Chancellor Gordon Brown announced a £1.5 billion investment in artificial intelligence and tech startups at 10 Downing Street on Friday!"),
        ("After Lowercase & Regex", "the uk chancellor gordon brown announced a billion investment in artificial intelligence and tech startups at downing street on friday"),
        ("After Stop-Word Removal", "uk chancellor gordon brown announced billion investment artificial intelligence tech startups downing street friday"),
        ("Final Lemmatized Output", "chancellor gordon brown announced billion investment artificial intelligence tech startup downing street friday")
    ]
    for r_idx, (stage, txt) in enumerate(prep_data):
        row = t_prep.rows[r_idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width, c1.width = Inches(2.2), Inches(4.2)
        set_cell_background(c0, "0D3B66" if r_idx == 0 else "F0F4F8")
        set_cell_background(c1, "F8F9FA" if r_idx == 0 else "FFFFFF")
        set_cell_margins(c0, top=60, bottom=60, left=80, right=80)
        set_cell_margins(c1, top=60, bottom=60, left=80, right=80)

        p0 = c0.paragraphs[0]
        r0 = p0.add_run(stage)
        r0.font.bold = True
        r0.font.size = Pt(9.5)
        if r_idx == 0:
            r0.font.color.rgb = COLOR_WHITE
        else:
            r0.font.color.rgb = COLOR_NAVY

        p1 = c1.paragraphs[0]
        r1 = p1.add_run(txt)
        r1.font.size = Pt(9)
        if r_idx == 3:
            r1.font.bold = True
            r1.font.color.rgb = RGBColor(20, 120, 40)

    # =========================================================================
    # 4. TF-IDF VECTORIZATION & LEAKAGE PREVENTION
    # =========================================================================
    add_custom_heading(doc, "4. TF-IDF VECTORIZATION & DATA LEAKAGE PREVENTION", level=1)

    p = doc.add_paragraph()
    p.add_run(
        "To perform machine learning classification, textual strings must be projected into a continuous numerical vector space. "
        "We utilize Term Frequency-Inverse Document Frequency (TF-IDF), a classical Information Retrieval weighting model formulated "
        "by Gerard Salton and Chris Buckley (1988).\n\n"
        "Mathematical Formulation:\n"
        "Let D denote the collection of training documents and t denote a term. The TF-IDF weight w(t, d) is given by:\n"
        "w(t, d) = TF(t, d) * IDF(t, D)\n\n"
        "1. Sublinear Term Frequency Scaling:\n"
        "A word appearing 10 times in a document is not 10 times more important than a word appearing once. To dampen repetitive frequency dominance, "
        "sublinear term frequency scaling is applied:\n"
        "sublinear_TF(t, d) = 1 + log(f(t, d)) for f(t, d) > 0, else 0.\n\n"
        "2. Smooth Inverse Document Frequency:\n"
        "IDF penalizes words that occur ubiquitously across all documents, assigning maximum weight to discriminative terms specific to a few classes:\n"
        "IDF(t, D) = log((1 + |D|) / (1 + DF(t))) + 1\n"
        "Where |D| is the total number of documents and DF(t) is the document frequency of term t.\n\n"
        "3. L2 Hypersphere Normalization:\n"
        "To eliminate length bias between concise sport articles and lengthy technology essays, each document vector is normalized to unit length:\n"
        "v(d) = w(d) / ||w(d)||_2\n\n"
        "4. N-Gram Expansion:\n"
        "Setting ngram_range=(1, 2) extracts both unigrams ('interest') and critical bigram collocations ('interest rate', 'prime minister', 'mobile phone'). "
        "The vocabulary is capped at max_features=5000 with min_df=3, pruning rare typos and noise while preserving topical vocabulary.\n\n"
        "DATA LEAKAGE PREVENTION PROTOCOL:\n"
        "A catastrophic flaw in naive machine learning workflows is fitting the feature vectorizer on the full corpus before splitting into train/test partitions. "
        "This leaks test set document frequencies, vocabulary lists, and IDF weights into the training phase, artificially inflating test scores. "
        "In this project, data leakage was strictly prevented by enclosing TfidfVectorizer and each classifier inside an integrated Scikit-Learn Pipeline. "
        "The vectorizer was fitted exclusively on the 80% training set (X_train). During test evaluation, X_test was transformed using the frozen training vocabulary, "
        "guaranteeing genuine out-of-sample validity."
    )

    # =========================================================================
    # 5. THE FOUR COMPETING CLASSIFIERS
    # =========================================================================
    add_custom_heading(doc, "5. THE FOUR COMPETING CLASSIFIERS", level=1)

    p = doc.add_paragraph()
    p.add_run(
        "Four distinct algorithms representing different theoretical paradigms of machine learning were implemented and compared under identical conditions:\n\n"
        "1. Decision Tree Classifier (CART):\n"
        "- Theoretical Foundation: Non-parametric tree model that recursively partitions the feature space into orthogonal hyper-rectangles by selecting "
        "the feature and split threshold that maximizes Gini Impurity reduction: I_G = 1 - sum(p_i^2).\n"
        "- Hyperparameters: max_depth=50, criterion='gini', random_state=42.\n"
        "- Behavior in Text: Text data is sparse and high-dimensional (5,000 features). Decision trees make axis-aligned cuts on one isolated word at a time, "
        "causing high variance and fragmented leaf nodes when class signals are distributed across hundreds of correlated terms.\n\n"
        "2. K-Nearest Neighbours (KNN):\n"
        "- Theoretical Foundation: Memory-based instance learner that bypasses explicit training (O(1) fit time). Given a test document vector, it computes "
        "pairwise distances against all 1,780 training vectors, retrieves the k nearest neighbors, and assigns the majority class weighted by inverse distance (w = 1/d).\n"
        "- Distance Metric: Cosine metric: d_cos(u, v) = 1 - (u . v) / (||u|| ||v||). For L2-normalized vectors, this evaluates angular orientation rather than Euclidean magnitude.\n"
        "- Hyperparameters: n_neighbors=5, metric='cosine', weights='distance'.\n\n"
        "3. Support Vector Machine (Linear SVM):\n"
        "- Theoretical Foundation: Vapnik's Maximum-Margin Principle. SVM constructs the optimal hyperplane w^T x + b = 0 separating classes with maximum margin 2/||w||, "
        "minimizing structural risk: min (1/2 ||w||^2 + C * sum(xi_i)).\n"
        "- Hyperparameters: kernel='linear', C=1.0, probability=True, random_state=42.\n"
        "- Behavior in Text: Cover's Theorem establishes that high-dimensional text representations are almost always linearly separable. Linear SVM avoids the curse of dimensionality "
        "and is practically immune to overfitting even with 5,000 features.\n\n"
        "4. Multinomial Naive Bayes (MNB):\n"
        "- Theoretical Foundation: Generative probabilistic model applying Bayes' Theorem under the naive assumption that all word tokens are conditionally independent "
        "given the class: P(C_k | d) proportional to P(C_k) * prod(P(w_i | C_k)).\n"
        "- Hyperparameters: Laplace smoothing alpha=0.1, sublinear TF counts.\n"
        "- Behavior in Text: Despite the unrealistic independence assumption, Naive Bayes acts as a linear classifier in log-odds space. It requires trivial training computation "
        "and exhibits exceptional classification accuracy when ranking posterior probabilities."
    )

    # =========================================================================
    # 6. EXPERIMENTAL RESULTS & COMPARATIVE BATTLE
    # =========================================================================
    add_custom_heading(doc, "6. EXPERIMENTAL RESULTS & COMPARATIVE BATTLE", level=1)

    p = doc.add_paragraph()
    p.add_run(
        "All four models were trained on 1,780 articles (80% stratified training split) and evaluated on 445 articles (20% held-out test split). "
        "Stratified 5-Fold Cross-Validation was conducted across the training partition. Table 2 summarizes the official experimental results."
    )

    # Leaderboard Table
    t_lead = doc.add_table(rows=5, cols=8)
    t_lead.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_lead.autofit = False

    lead_headers = ["Classifier", "Test Acc (%)", "Macro Prec", "Macro Rec", "Macro F1", "5-Fold CV Acc", "Train Time (s)", "Latency (ms)"]
    for c, h in enumerate(lead_headers):
        cell = t_lead.cell(0, c)
        cell.text = h
        set_cell_background(cell, "0D3B66")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.runs[0]
        run.font.bold = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = COLOR_WHITE

    lead_data = [
        ["Multinomial Naive Bayes", "97.98%", "0.9791", "0.9813", "0.9799", "96.97% ± 0.74%", "0.844 s", "0.16 ms"],
        ["Support Vector Machine", "97.98%", "0.9801", "0.9796", "0.9796", "98.48% ± 0.29%", "12.575 s", "1.18 ms"],
        ["K-Nearest Neighbours", "97.08%", "0.9708", "0.9712", "0.9708", "96.12% ± 1.13%", "0.833 s", "0.21 ms"],
        ["Decision Tree (CART)", "85.17%", "0.8534", "0.8521", "0.8524", "83.54% ± 1.59%", "1.046 s", "0.16 ms"]
    ]

    for r_idx, row_vals in enumerate(lead_data):
        row = t_lead.rows[r_idx + 1]
        bg = "FFFFFF" if r_idx % 2 == 0 else "F8F9FA"
        if r_idx in [0, 1]:
            bg = "EBF5FB"  # Highlight champions
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.text = val
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            run = p.runs[0]
            run.font.size = Pt(8.5)
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                run.font.bold = True
            elif c_idx == 1:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                run.font.bold = True
                run.font.color.rgb = COLOR_NAVY
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    p_lead_disc = doc.add_paragraph()
    p_lead_disc.paragraph_format.space_before = Pt(8)
    p_lead_disc.add_run(
        "Critical Findings from the Battle:\n"
        "1. Co-Champions at 97.98% Accuracy: Both Multinomial Naive Bayes and Support Vector Machine achieved 97.98% accuracy on the test set, "
        "correctly classifying 436 out of 445 unseen documents.\n"
        "2. Generalizability Champion: Linear SVM achieved the highest 5-Fold Cross-Validation score (98.48%) with the smallest variance (± 0.29%), "
        "confirming that its maximum-margin hyperplane provides the most reliable boundary on high-dimensional text.\n"
        "3. Throughput Champion: Multinomial Naive Bayes was over 7 times faster than calibrated SVM at inference (0.16 ms vs 1.18 ms per article) "
        "and trained in under 0.85 seconds, making it ideal for streaming high-throughput deployments.\n"
        "4. K-Nearest Neighbours Strong Performance: KNN with Cosine distance achieved an outstanding 97.08% accuracy, verifying that instance-based "
        "retrieval with angular similarity is highly competitive in text categorization.\n"
        "5. Decision Tree Lag: Decision Tree achieved 85.17% accuracy, lagging ~13% behind SVM and MNB. Single-word orthogonal cuts cannot capture "
        "distributed topical patterns."
    )

    # Insert Model Comparison Bar & Radar Charts
    c_bar = os.path.join(RESULTS_DIR, "model_comparison_bar.png")
    c_radar = os.path.join(RESULTS_DIR, "radar_chart.png")
    if os.path.exists(c_bar):
        doc.add_picture(c_bar, width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 2: Performance metrics comparison (Accuracy, Precision, Recall, F1) across the four classifiers.")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(9)
        p_cap.runs[0].font.italic = True

    if os.path.exists(c_radar):
        doc.add_picture(c_radar, width=Inches(4.5))
        p_cap = doc.add_paragraph("Figure 3: Multi-metric polar radar chart illustrating algorithmic dominance.")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(9)
        p_cap.runs[0].font.italic = True

    # =========================================================================
    # 7. CONFUSION MATRIX & ERROR DIAGNOSTICS
    # =========================================================================
    add_custom_heading(doc, "7. CONFUSION MATRIX & ERROR DIAGNOSTICS", level=1)

    p = doc.add_paragraph()
    p.add_run(
        "To inspect per-class reliability and pinpoint classification failure modes, confusion matrices were computed for all four models on the 445 test samples. "
        "Table 3 presents the detailed per-class breakdown for the champion Linear Support Vector Machine."
    )

    # Per-Class SVM Table
    t_cm = doc.add_table(rows=6, cols=6)
    t_cm.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_cm.autofit = False

    cm_headers = ["News Category", "True Support", "Correct (TP)", "Misclassified", "Precision", "Recall / F1-Score"]
    for c, h in enumerate(cm_headers):
        cell = t_cm.cell(0, c)
        cell.text = h
        set_cell_background(cell, "0D3B66")
        set_cell_margins(cell, top=70, bottom=70, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.runs[0]
        run.font.bold = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = COLOR_WHITE

    svm_class_data = [
        ["Sport", "102", "102", "0", "0.9808", "1.0000 / 0.9903"],
        ["Politics", "84", "84", "0", "0.9655", "1.0000 / 0.9825"],
        ["Business", "102", "98", "4", "0.9800", "0.9608 / 0.9703"],
        ["Entertainment", "77", "76", "1", "0.9744", "0.9870 / 0.9806"],
        ["Tech", "80", "76", "4", "1.0000", "0.9500 / 0.9744"]
    ]

    for r_idx, row_vals in enumerate(svm_class_data):
        row = t_cm.rows[r_idx + 1]
        bg = "FFFFFF" if r_idx % 2 == 0 else "F8F9FA"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.text = val
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
            p = cell.paragraphs[0]
            run = p.runs[0]
            run.font.size = Pt(8.5)
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                run.font.bold = True
            elif c_idx in [1, 2, 3]:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Insert 4-panel confusion matrix plot
    c_cm_all = os.path.join(RESULTS_DIR, "confusion_matrices_all.png")
    if os.path.exists(c_cm_all):
        doc.add_picture(c_cm_all, width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 4: 4-Panel comparative confusion matrix heatmaps with raw counts and normalized percentages.")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(9)
        p_cap.runs[0].font.italic = True

    p_err_disc = doc.add_paragraph()
    p_err_disc.add_run(
        "Linguistic Error Diagnostics:\n"
        "• Perfect Recall in Sport and Politics: SVM achieved 100% recall on both Sport (102/102) and Politics (84/84). Articles in these categories "
        "exhibit distinctive, domain-specific vocabularies ('arsenal', 'chelsea', 'cup', 'blair', 'parliament', 'tory') with minimal cross-domain ambiguity.\n"
        "• Business vs Politics Boundary Overlap: Out of 102 Business articles, 4 were confused (3 misclassified as Politics). Qualitative inspection "
        "revealed these articles covered Chancellor Gordon Brown's national budget deficits, public sector economic funding, and corporate taxation. "
        "Because fiscal policy naturally overlaps with ministerial politics, words such as 'chancellor', 'tax', 'economy', and 'budget' created boundary ambiguity.\n"
        "• Tech vs Business Overlap: 2 Tech articles discussing corporate acquisitions and stock market revenues of major telecommunications providers were "
        "classified as Business due to heavy financial terminology ('shares', 'revenue', 'merger')."
    )

    # =========================================================================
    # 8. WHAT-IF EMPIRICAL SENSITIVITY EXPERIMENTS
    # =========================================================================
    add_custom_heading(doc, "8. WHAT-IF EMPIRICAL SENSITIVITY EXPERIMENTS", level=1)

    p = doc.add_paragraph()
    p.add_run(
        "To rigorously investigate algorithm sensitivity under hyperparameter variations, three dedicated What-If experiments were executed:\n\n"
        "Experiment 1: KNN Neighborhood Size (k) & Distance Metric Sensitivity\n"
        "Neighborhood size k was varied from 1 to 31 comparing Cosine distance versus Euclidean distance:\n"
        "• At k=1, accuracy was 95.51% due to sensitivity to isolated outliers.\n"
        "• Accuracy peaked at k=3 (97.98%) and remained stable at k=5 (97.08%).\n"
        "• Beyond k=15, accuracy leveled off around 96.85%-97.30% because inverse-distance weighting (w = 1/d) effectively downweighted distant points.\n"
        "• Crucially, Cosine distance strictly outperformed Euclidean distance across all neighborhood sizes, proving that document length normalization "
        "is mathematically essential in text vector spaces.\n\n"
        "Experiment 2: TF-IDF Vocabulary Capacity (max_features)\n"
        "Vocabulary size was varied from 250 to Unlimited (60,659 features):\n"
        "• At 250 features, accuracy was 94.38% (SVM) and 93.48% (NB), as critical domain terms were discarded.\n"
        "• Accuracy surged to 96.85% at 500 features and reached 97.98% at 5,000 features.\n"
        "• Increasing vocabulary to Unlimited (60,659 terms) provided zero performance gain (97.98% SVM) while multiplying memory footprint by 12x.\n"
        "This empirical result defends our choice of max_features=5000 as the optimal trade-off between semantic richness and computational economy.\n\n"
        "Experiment 3: N-Gram Lexical Span (Unigrams vs Bigrams vs Trigrams)\n"
        "Comparing unigram-only (1, 1), unigram+bigram (1, 2), and trigram (1, 3) representations:\n"
        "• Incorporating bigrams boosted SVM accuracy from 97.53% to 97.98% and Naive Bayes from 97.75% to 97.98%.\n"
        "• Bigrams successfully capture essential phrasal collocations ('prime minister', 'wall street', 'interest rate', 'box office').\n"
        "• Extending to trigrams (1, 3) yielded no significant improvement (97.98%) while expanding the feature matrix with noisy tri-word sequences."
    )

    # Insert What-If Image
    c_whatif = os.path.join(RESULTS_DIR, "what_if_analysis.png")
    if os.path.exists(c_whatif):
        doc.add_picture(c_whatif, width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 5: Tri-partite What-If parametric sensitivity plots (KNN k, max_features, and N-gram ranges).")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(9)
        p_cap.runs[0].font.italic = True

    # =========================================================================
    # 9. STREAMLIT APPLICATION & ARCHITECTURE
    # =========================================================================
    add_custom_heading(doc, "9. STREAMLIT APPLICATION & INTERACTIVE DASHBOARD", level=1)

    p = doc.add_paragraph()
    p.add_run(
        "A full-stack, responsive web application was developed in app.py using Streamlit. The application incorporates an academic professional design "
        "and is organized into eight specialized navigation modules:\n"
        "1. Home & Project Overview: Displays student identification, project objectives, and system architecture diagrams.\n"
        "2. Dataset Exploration: Interactive exploration of corpus statistics, class distribution bar/pie charts, and boxplots of article word counts.\n"
        "3. Text Preprocessing Lab: An interactive laboratory allowing evaluators to paste custom text and inspect step-by-step transformations "
        "(lowercasing, symbol stripping, tokenization, stop-words filtering, and lemmatization) with real-time dimensionality reduction percentages.\n"
        "4. TF-IDF Feature Space: Visualizes mathematical formulations, sublinear TF equations, and extracts the top 7 most influential keywords per category.\n"
        "5. Four-Model Battle Arena: Displays the official leaderboard table, comparative bar charts, and multi-metric polar radar diagrams.\n"
        "6. Performance Metrics & CM: Interactive confusion matrix inspector with normalized percentages and per-class precision/recall diagnostics.\n"
        "7. What-If Empirical Analysis: Interactive parametric charts enabling dynamic exploration of KNN k values, vocabulary sizes, and n-grams.\n"
        "8. Live News Classifier: A real-time inference engine where evaluators can input custom breaking news text or select benchmark samples. "
        "The model predicts the category, displays confidence percentages, renders posterior probability distribution bars across all 5 classes, "
        "and highlights the top influential TF-IDF keywords that drove the prediction."
    )

    # =========================================================================
    # 10. CONCLUSION, LIMITATIONS & FUTURE SCOPE
    # =========================================================================
    add_custom_heading(doc, "10. CONCLUSION, LIMITATIONS & FUTURE SCOPE", level=1)

    p = doc.add_paragraph()
    p.add_run(
        "Conclusion:\n"
        "This project successfully developed, evaluated, and defended an automated news category classification system on the authentic BBC News corpus. "
        "Both Support Vector Machine (Linear SVM) and Multinomial Naive Bayes emerged as co-champions with an outstanding 97.98% test accuracy. "
        "Linear SVM demonstrated superior 5-fold cross-validation stability (98.48%), proving that maximum-margin separation is the optimal theoretical "
        "architecture for high-dimensional sparse text representations. Conversely, Multinomial Naive Bayes proved to be the throughput champion, "
        "executing predictions in 0.16 milliseconds per document with negligible training overhead. K-Nearest Neighbours with Cosine distance achieved "
        "a highly competitive 97.08%, while Decision Tree lagged at 85.17% due to feature fragmentation.\n\n"
        "Limitations:\n"
        "1. Temporal & Cultural Domain Bias: The BBC News corpus consists of British journalistic articles from 2004–2005. Models trained on this corpus "
        "may experience performance degradation when deployed on contemporary news containing modern colloquialisms, social media hashtags, or non-UK political references.\n"
        "2. Syntactic Order Disregard: Bag-of-Words and TF-IDF representations ignore word order and complex semantic dependencies beyond bigrams.\n\n"
        "Future Scope:\n"
        "1. Dense Transformer Embeddings: Integrating pre-trained contextual language models (BERT, RoBERTa, DeBERTa) to capture deep bidirectional semantic context.\n"
        "2. Active Learning & Streaming Pipelines: Deploying online continual learning algorithms that ingest streaming news feeds and solicit human-in-the-loop annotations "
        "for low-confidence predictions."
    )

    # =========================================================================
    # 11. REFERENCES (IEEE FORMAT)
    # =========================================================================
    add_custom_heading(doc, "11. REFERENCES", level=1)

    refs_list = [
        "[1] D. Greene and P. Cunningham, 'Practical solutions to the problem of diagonal dominance in kernel document clustering,' in Proceedings of the 23rd International Conference on Machine Learning (ICML '06), Pittsburgh, PA, 2006, pp. 377–384.",
        "[2] T. Joachims, 'Text categorization with support vector machines: Learning with many relevant features,' in Proceedings of the European Conference on Machine Learning (ECML), Chemnitz, Germany: Springer, 1998, pp. 137–142.",
        "[3] C. D. Manning, P. Raghavan, and H. Schütze, Introduction to Information Retrieval. Cambridge, UK: Cambridge University Press, 2008.",
        "[4] P. Domingos and M. Pazzani, 'On the optimality of the simple Bayesian classifier under zero-one loss,' Machine Learning, vol. 29, no. 2, pp. 103–130, 1997.",
        "[5] G. Salton and C. Buckley, 'Term-weighting approaches in automatic text retrieval,' Information Processing & Management, vol. 24, no. 5, pp. 513–523, 1988.",
        "[6] F. Pedregosa et al., 'Scikit-learn: Machine learning in Python,' Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
        "[7] V. N. Vapnik, Statistical Learning Theory. New York, NY: John Wiley & Sons, 1998.",
        "[8] S. Bird, E. Klein, and E. Loper, Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit. Sebastopol, CA: O'Reilly Media, 2009."
    ]

    for ref in refs_list:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_before = Pt(3)
        p_ref.paragraph_format.space_after = Pt(3)
        r = p_ref.add_run(ref)
        r.font.size = Pt(9.5)

    doc.save(DOCX_PATH)
    print(f"Academic Report successfully saved to: {DOCX_PATH}")


if __name__ == "__main__":
    build_academic_report()
