"""RESQUER AI experimental NLP demo API. Not a diagnostic or crisis prediction tool."""
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.features import extract_features
from app.model import model

app = FastAPI(
    title="RESQUER AI — Experimental NLP Demo",
    description=("Research prototype for exploring text features. Outputs are uncertain, based on a tiny synthetic demo corpus, and must never be used as a diagnosis or as the sole basis for intervention. Do not submit identifiable or private posts."),
    version="0.1.0",
)

class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000, description="Non-identifying text you have permission to analyze")
    consent_confirmed: bool = Field(default=False, description="Confirm you have permission to analyze this text")

class AnalyzeResponse(BaseModel):
    status: Literal["experimental_demo"]
    features: dict
    demo_model_score: float
    interpretation: str
    limitations: list[str]

@app.get("/")
def root():
    return {"name": "RESQUER AI", "status": "experimental_demo", "docs": "/docs", "warning": "Not for diagnosis or real-world risk decisions."}

@app.get("/health")
def health():
    return {"status": "ok", "model": "synthetic_tfidf_logistic_regression_demo"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    text = request.text.strip()
    if not request.consent_confirmed:
        raise HTTPException(status_code=400, detail="Confirm that you have permission to analyze this text.")
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be blank.")
    features = extract_features(text)
    # This is a toy-model output, not a probability of suicide or harm.
    score = float(model.predict_proba([text])[0][1])
    return AnalyzeResponse(
        status="experimental_demo",
        features=features,
        demo_model_score=round(score, 4),
        interpretation="Toy model signal only. It cannot determine whether a person is suicidal or safe.",
        limitations=[
            "Trained on ten synthetic examples; not representative or clinically validated.",
            "Text-only analysis misses context, culture, sarcasm, and individual circumstances.",
            "Do not use this score to diagnose, surveil people, or automatically contact third parties.",
            "For real research, use governed consented data, expert labels, subgroup evaluation, and human review.",
        ],
    )
