"""
Inference and Real-Time Prediction Module
Author: Sathish Kumar R (RA2411026050188)
Institution: SRM Institute of Science and Technology, Tiruchirappalli
Course: 21CSE377T - Information Retrieval
Assignment: CLA-2 Task 2
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import TextPreprocessor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class NewsPredictor:
    """
    Inference engine that loads trained pipeline artifacts and performs
    category classification with confidence scores and top influential keywords.
    """

    def __init__(self, model_name: str = "best_model"):
        self.preprocessor = TextPreprocessor(use_stemmer=False, use_lemmatizer=True)
        self.model_name = model_name
        self.pipeline = self._load_pipeline(model_name)

    def _load_pipeline(self, model_name: str):
        filename_map = {
            "best_model": "best_model.joblib",
            "Support Vector Machine": "support_vector_machine_pipeline.joblib",
            "Multinomial Naive Bayes": "multinomial_naive_bayes_pipeline.joblib",
            "K-Nearest Neighbours": "k_nearest_neighbours_pipeline.joblib",
            "Decision Tree": "decision_tree_pipeline.joblib"
        }
        filename = filename_map.get(model_name, f"{model_name.lower().replace(' ', '_').replace('-', '_')}_pipeline.joblib")
        model_path = os.path.join(MODELS_DIR, filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train models first using model_training.py")
        
        return joblib.load(model_path)

    def predict(self, raw_text: str) -> dict:
        """
        Processes raw text and predicts the news category.

        Parameters:
            raw_text (str): Raw article headline or full text

        Returns:
            dict: {
                'predicted_category': str,
                'confidence': float,
                'probabilities': dict,
                'cleaned_text': str,
                'top_keywords': list[tuple[str, float]]
            }
        """
        if not raw_text or not raw_text.strip():
            return {
                'predicted_category': 'Unknown',
                'confidence': 0.0,
                'probabilities': {},
                'cleaned_text': '',
                'top_keywords': []
            }

        cleaned = self.preprocessor.clean_text(raw_text, method='lemmatize')
        if not cleaned:
            cleaned = raw_text.lower().strip()

        # Model prediction
        pred_label = self.pipeline.predict([cleaned])[0]

        # Extract confidence / probabilities
        classes = self.pipeline.classes_
        probabilities = {}
        confidence = 1.0

        if hasattr(self.pipeline.named_steps['clf'], "predict_proba"):
            probs = self.pipeline.predict_proba([cleaned])[0]
            for cls_name, prob in zip(classes, probs):
                probabilities[cls_name] = round(float(prob), 4)
            confidence = probabilities[pred_label]
        elif hasattr(self.pipeline.named_steps['clf'], "decision_function"):
            scores = self.pipeline.decision_function([cleaned])[0]
            # Softmax on decision scores
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)
            for cls_name, prob in zip(classes, probs):
                probabilities[cls_name] = round(float(prob), 4)
            confidence = probabilities[pred_label]
        else:
            # Fallback for models without probabilistic estimation
            for cls_name in classes:
                probabilities[cls_name] = 1.0 if cls_name == pred_label else 0.0

        # Extract top driving TF-IDF keywords
        tfidf_step = self.pipeline.named_steps['tfidf']
        vec = tfidf_step.transform([cleaned])
        feature_names = tfidf_step.get_feature_names_out()
        coo = vec.tocoo()
        keyword_scores = sorted(zip(coo.col, coo.data), key=lambda x: x[1], reverse=True)[:8]
        top_keywords = [(feature_names[idx], round(float(score), 4)) for idx, score in keyword_scores]

        return {
            'predicted_category': pred_label,
            'confidence': confidence,
            'probabilities': probabilities,
            'cleaned_text': cleaned,
            'top_keywords': top_keywords
        }

    def predict_batch(self, texts: list[str]) -> pd.DataFrame:
        """
        Batch prediction helper for testing multiple news headlines.
        """
        records = []
        for text in texts:
            res = self.predict(text)
            records.append({
                'Input Text': text[:80] + "..." if len(text) > 80 else text,
                'Predicted Category': res['predicted_category'],
                'Confidence': f"{res['confidence']*100:.1f}%",
                'Top Keywords': ", ".join([w for w, s in res['top_keywords'][:4]])
            })
        return pd.DataFrame(records)


if __name__ == "__main__":
    predictor = NewsPredictor(model_name="best_model")
    sample_news = [
        "Manchester United secured a dramatic 2-1 victory over Chelsea in the Premier League derby yesterday.",
        "The Federal Reserve raised benchmark interest rates by 25 basis points to combat rising inflation and stabilize bond markets.",
        "Apple unveiled its new M3 chip architecture with high-speed neural engine capabilities for artificial intelligence workstations.",
        "The Prime Minister faced intense scrutiny during Parliament Question Time over public sector healthcare funding and election pledges.",
        "The Hollywood musical won six Academy Awards at the glamorous Oscars ceremony in Los Angeles on Sunday night."
    ]
    df_preds = predictor.predict_batch(sample_news)
    print("\n=== Live Sample Predictions ===")
    print(df_preds.to_string(index=False))

