import pandas as pd
from collections import Counter
from datetime import datetime, timedelta

# --- Anomaly Detection Functions ---
def detect_ticket_volume_anomalies(tickets, window_days=7, threshold=2.0):
    """Detects categories with unusual ticket volume spikes in the last window_days."""
    tickets['created_at'] = pd.to_datetime(tickets['created_at'])
    recent = tickets[tickets['created_at'] > datetime.now() - timedelta(days=window_days)]
    volume = recent.groupby('category').size()
    mean = tickets.groupby('category').size().mean()
    anomalies = volume[volume > threshold * mean].to_dict()
    return anomalies

def detect_new_issue_types(tickets, known_categories):
    """Detects tickets with categories not in known_categories."""
    new_issues = tickets[~tickets['category'].isin(known_categories)]
    return new_issues[['ticket_id', 'category', 'subject', 'description']].to_dict(orient='records')

def detect_sentiment_shifts(tickets, product_col='product', sentiment_col='customer_sentiment', window_days=30):
    """Detects sentiment shifts by product in the last window_days."""
    tickets['created_at'] = pd.to_datetime(tickets['created_at'])
    recent = tickets[tickets['created_at'] > datetime.now() - timedelta(days=window_days)]
    sentiment_counts = recent.groupby([product_col, sentiment_col]).size().unstack(fill_value=0)
    return sentiment_counts.to_dict()

def detect_retrieval_failures(retrieval_logs):
    """Detects queries with no solutions returned (retrieval failures)."""
    failures = [log for log in retrieval_logs if not log.get('solutions')]
    return failures

# Example usage (to be called from API or notebook):
# tickets = pd.read_json('support_tickets_sample.json')
# anomalies = detect_ticket_volume_anomalies(tickets)
# new_issues = detect_new_issue_types(tickets, known_categories=[...])
# sentiment = detect_sentiment_shifts(tickets)
# failures = detect_retrieval_failures(retrieval_logs)
