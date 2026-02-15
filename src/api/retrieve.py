from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os

router = APIRouter()

# Dummy: Load historical tickets for retrieval (in production, use a DB or vector store)
TICKETS_PATH = os.getenv("TICKETS_PATH", "support_tickets.json")
if os.path.exists(TICKETS_PATH):
    with open(TICKETS_PATH, "r") as f:
        tickets = pd.read_json(f)
else:
    tickets = pd.DataFrame()

class RetrievalInput(BaseModel):
    category: str
    subject: str
    description: str
    # Add more fields as needed

@router.post("/retrieve")
def retrieve_solutions(query: RetrievalInput):
    # Filter tickets by predicted category
    filtered = tickets[tickets["category"] == query.category] if not tickets.empty else pd.DataFrame()
    # Simple keyword match in subject/description (demo)
    if not filtered.empty:
        mask = filtered["subject"].str.contains(query.subject, case=False, na=False) | \
               filtered["description"].str.contains(query.description, case=False, na=False)
        results = filtered[mask].head(5)
        solutions = results["resolution"].tolist()
    else:
        solutions = []
    return {"solutions": solutions}
