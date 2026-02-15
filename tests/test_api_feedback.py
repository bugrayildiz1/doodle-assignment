import requests

API_URL = "http://127.0.0.1:8000/feedback"

def test_feedback():
    payload = {
        "ticket_id": "TK-2026-000001",
        "agent_id": "AGENT-001",
        "correction": "Category should be 'Billing', not 'Technical'.",
        "customer_id": "CUST-12345",
        "feedback_text": "Agent was helpful, but initial answer was wrong.",
        "satisfaction_score": 4
    }
    print("Sending feedback to /feedback...")
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        print("Status Code:", response.status_code)
        try:
            print("Response JSON:", response.json())
        except Exception:
            print("Response Text:", response.text)
    except Exception as e:
        import traceback
        print("Exception occurred:", e)
        traceback.print_exc()
    print("Feedback test completed.")

if __name__ == "__main__":
    test_feedback()
