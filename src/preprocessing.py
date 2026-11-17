"""
Text Preprocessing Module for Information Retrieval & Text Classification
Author: Sathish Kumar R (RA2411026050188)
Institution: SRM Institute of Science and Technology, Tiruchirappalli
Course: 21CSE377T - Information Retrieval
Assignment: CLA-2 Task 2
"""

import os
import re
import string
import nltk

# Configure local nltk_data search paths so it runs entirely self-contained
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
_LOCAL_NLTK_DATA = os.path.join(_PROJECT_ROOT, "nltk_data")
if os.path.exists(_LOCAL_NLTK_DATA) and _LOCAL_NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, _LOCAL_NLTK_DATA)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline implementing:
    1. Lowercase normalization
    2. URL & HTML tag removal
    3. Punctuation and digit removal
    4. Tokenization
    5. Stop-word removal (NLTK English stop words)
    6. Stemming (Porter Stemmer) or Lemmatization (WordNet Lemmatizer)
    """

    def __init__(self, use_stemmer: bool = False, use_lemmatizer: bool = True):
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            # Fallback english stopwords if corpus loading has issues
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
                'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
                'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
                'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
                'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
                'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
                'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
            }
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.use_stemmer = use_stemmer
        self.use_lemmatizer = use_lemmatizer

    def tokenize_text(self, text: str) -> list[str]:
        """Tokenize text using NLTK word_tokenize with regex fallback."""
        try:
            return word_tokenize(text)
        except Exception:
            return re.findall(r'\b[a-z]{2,}\b', text)

    def clean_text(self, text: str, method: str = 'lemmatize') -> str:
        """
        Cleans and normalizes raw text into preprocessed tokens joined by spaces.

        Parameters:
            text (str): Raw input news text
            method (str): 'lemmatize', 'stem', or 'none'

        Returns:
            str: Preprocessed text string
        """
        if not isinstance(text, str):
            return ""

        # Step 1: Lowercase conversion
        text_lower = text.lower()

        # Step 2: Remove URLs and web links
        text_no_url = re.sub(r'http\S+|www\.\S+', ' ', text_lower)

        # Step 3: Remove HTML tags
        text_no_html = re.sub(r'<.*?>', ' ', text_no_url)

        # Step 4: Keep only alphabetic characters (removes digits, punctuation, special symbols)
        text_alpha = re.sub(r'[^a-z\s]', ' ', text_no_html)

        # Step 5: Normalize extra whitespaces
        text_clean = re.sub(r'\s+', ' ', text_alpha).strip()

        # Step 6: Tokenization
        tokens = self.tokenize_text(text_clean)

        # Step 7: Stop-word removal (filtering tokens of length >= 2 and not in stopwords)
        filtered_tokens = [tok for tok in tokens if tok not in self.stop_words and len(tok) > 2]

        # Step 8: Morphological normalization (Stemming or Lemmatization)
        if method == 'stem' or (self.use_stemmer and method != 'lemmatize'):
            normalized_tokens = [self.stemmer.stem(tok) for tok in filtered_tokens]
        elif method == 'lemmatize' or self.use_lemmatizer:
            try:
                normalized_tokens = [self.lemmatizer.lemmatize(tok) for tok in filtered_tokens]
            except Exception:
                normalized_tokens = [self.stemmer.stem(tok) for tok in filtered_tokens]
        else:
            normalized_tokens = filtered_tokens

        return " ".join(normalized_tokens)

    def demonstrate_steps(self, text: str) -> dict:
        """
        Executes and captures every intermediate stage of preprocessing
        for pedagogical inspection, reporting, and Streamlit visualization.

        Parameters:
            text (str): Raw input sample text

        Returns:
            dict: Step-by-step intermediate representations
        """
        step1_lower = text.lower()
        step2_no_url = re.sub(r'http\S+|www\.\S+', ' ', step1_lower)
        step3_alpha = re.sub(r'[^a-z\s]', ' ', step2_no_url)
        step4_tokens = self.tokenize_text(step3_alpha)
        step5_no_stop = [tok for tok in step4_tokens if tok not in self.stop_words and len(tok) > 2]
        step6_stem = [self.stemmer.stem(tok) for tok in step5_no_stop]
        try:
            step7_lemma = [self.lemmatizer.lemmatize(tok) for tok in step5_no_stop]
        except Exception:
            step7_lemma = step6_stem

        return {
            "1_raw_text": text,
            "2_lowercase": step1_lower,
            "3_removed_punct_digits": step3_alpha.strip(),
            "4_tokens": step4_tokens,
            "4_token_count": len(step4_tokens),
            "5_stopwords_removed": step5_no_stop,
            "5_filtered_token_count": len(step5_no_stop),
            "6_stemmed_tokens": step6_stem,
            "6_stemmed_text": " ".join(step6_stem),
            "7_lemmatized_tokens": step7_lemma,
            "7_lemmatized_text": " ".join(step7_lemma),
            "token_reduction_pct": round((1 - len(step7_lemma) / max(len(step4_tokens), 1)) * 100, 2)
        }


def get_default_preprocessor() -> TextPreprocessor:
    """Helper factory for default preprocessor instance."""
    return TextPreprocessor(use_stemmer=False, use_lemmatizer=True)


if __name__ == "__main__":
    preprocessor = get_default_preprocessor()
    sample = "The UK Chancellor Gordon Brown announced a £1.5 billion investment in artificial intelligence and tech startups at 10 Downing Street on Friday!"
    print("=== Raw Text ===")
    print(sample)
    print("\n=== Preprocessed (Lemmatized) ===")
    print(preprocessor.clean_text(sample, method='lemmatize'))
    print("\n=== Preprocessed (Stemmed) ===")
    print(preprocessor.clean_text(sample, method='stem'))

    demo = preprocessor.demonstrate_steps(sample)
    print("\n=== Step-by-Step Breakdown ===")
    print(f"Raw Token Count: {demo['4_token_count']}")
    print(f"Filtered Token Count: {demo['5_filtered_token_count']}")
    print(f"Dimensionality / Token Reduction: {demo['token_reduction_pct']}%")
