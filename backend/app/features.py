"""Transparent, lightweight English text features for the RESQUER AI demo."""
import re
from collections import Counter
from textblob import TextBlob

TOKEN_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
SELF_REFERENCE = {"i", "me", "my", "mine", "myself"}
NEGATIVE_CUES = {"hopeless", "worthless", "empty", "alone", "tired", "pain", "hurt", "trapped", "burden", "cannot", "goodbye", "disappear"}
SUPPORT_CUES = {"help", "support", "talk", "friend", "family", "counsellor", "counselor"}

def extract_features(text: str) -> dict:
    """Return interpretable text statistics; these are not clinical indicators."""
    clean = (text or "").strip()
    tokens = [t.lower() for t in TOKEN_RE.findall(clean)]
    counts = Counter(tokens)
    sentiment = TextBlob(clean).sentiment if clean else None
    return {
        "word_count": len(tokens),
        "character_count": len(clean),
        "self_reference_count": sum(counts[w] for w in SELF_REFERENCE),
        "negative_cue_count": sum(counts[w] for w in NEGATIVE_CUES),
        "support_cue_count": sum(counts[w] for w in SUPPORT_CUES),
        "sentiment_polarity": round(float(sentiment.polarity), 4) if sentiment else 0.0,
        "sentiment_subjectivity": round(float(sentiment.subjectivity), 4) if sentiment else 0.0,
    }
