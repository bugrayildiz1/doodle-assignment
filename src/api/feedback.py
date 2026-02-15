from fastapi import APIRouter
from pydantic import BaseModel
import json
import os
from datetime import datetime

router = APIRouter()

FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "feedback_log.json")

class FeedbackInput(BaseModel):
    ticket_id: str
    agent_id: str
    correction: str = None  # Agent correction (optional)
    customer_id: str = None
    feedback_text: str = None  # Customer feedback (optional)
    satisfaction_score: int = None  # 1-5 (optional)
    resolution_success: bool = None  # True if resolution was successful, False otherwise

@router.post("/feedback")
def capture_feedback(feedback: FeedbackInput):
    entry = feedback.dict()
    entry["timestamp"] = datetime.utcnow().isoformat()
    # Append to feedback_log.json
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return {"message": "Feedback/correction recorded", "entry": entry}
