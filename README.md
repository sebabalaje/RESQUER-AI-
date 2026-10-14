# RESQUER AI

RESQUER AI is an experimental NLP-based early-support system for identifying language patterns associated with emotional distress and possible self-harm risk.

> **Safety note:** RESQUER AI is not a suicide detector, diagnosis system, or emergency-response authority. Model outputs are risk signals for research/prototyping and must not be used as the sole basis for intervention.

## Current prototype
- English text preprocessing
- TF-IDF + Logistic Regression baseline classifier
- Sentiment/polarity features
- Linguistic self-reference features (I/me/my)
- First-person and lexical statistics
- FastAPI inference API
- Clear separation between model scoring and human-support decisions

## Privacy
Use only consented, public, or legally obtained data. Never upload private social-media posts, personal identifiers, credentials, or sensitive datasets to this repository.

## Run locally

### Backend
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

### Test
```bash
cd backend
python -m pytest
```

## Project status
The repository currently contains the safe baseline architecture. A validated labeled dataset is required before claiming meaningful real-world performance.
