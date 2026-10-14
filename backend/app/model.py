"""Toy baseline classifier trained only on illustrative synthetic examples.

Never use this model to assess real people. The tiny demo corpus is not representative
and cannot support clinical or real-world performance claims.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_DEMO_TEXTS = [
    "I feel great today and enjoyed time with my friends",
    "I am excited about my new project and future plans",
    "Had a nice day and I feel calm and supported",
    "I am looking forward to meeting my family tomorrow",
    "I feel hopeless and alone and nothing seems to help",
    "I feel worthless and trapped in pain",
    "I cannot cope today and I feel completely empty",
    "I am overwhelmed and feel like a burden to everyone",
    "I feel sad and isolated and need someone to talk to",
    "I am struggling and everything feels too much right now",
]
_DEMO_LABELS = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

def build_demo_model() -> Pipeline:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=3000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=7)),
    ])
    pipeline.fit(_DEMO_TEXTS, _DEMO_LABELS)
    return pipeline

model = build_demo_model()
