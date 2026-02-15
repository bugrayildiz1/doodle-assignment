from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import os
from src.analytics.anomaly_detection import (
    detect_ticket_volume_anomalies,
    detect_new_issue_types,
    detect_sentiment_shifts,
    detect_retrieval_failures
)

router = APIRouter()

class AnomalyRequest(BaseModel):
    known_categories: list[str] = []
    retrieval_logs: list = []  # List of dicts with 'solutions' key
    window_days: int = 7

@router.post("/analyze_anomalies")
def analyze_anomalies(req: AnomalyRequest):
    TICKETS_PATH = os.getenv("TICKETS_PATH", "support_tickets_sample.json")
    if os.path.exists(TICKETS_PATH):
        tickets = pd.read_json(TICKETS_PATH)
    else:
        return {"error": "Tickets file not found."}
    volume_anomalies = detect_ticket_volume_anomalies(tickets, window_days=req.window_days)
    new_issues = detect_new_issue_types(tickets, req.known_categories)
    sentiment = detect_sentiment_shifts(tickets, window_days=30)
    retrieval_failures = detect_retrieval_failures(req.retrieval_logs)
    return {
        "volume_anomalies": volume_anomalies,
        "new_issue_types": new_issues,
        "sentiment_shifts": sentiment,
        "retrieval_failures": retrieval_failures
    }
