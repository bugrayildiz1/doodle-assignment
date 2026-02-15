import requests

API_URL = "http://127.0.0.1:8000/feedback"

def test_feedback_resolution():
    payload = {
        "ticket_id": "TK-2026-000002",
        "agent_id": "AGENT-002",
        "correction": "Subcategory should be 'Login', not 'Account'.",
        "customer_id": "CUST-54321",
        "feedback_text": "Issue was not resolved on first attempt.",
        "satisfaction_score": 2,
        "resolution_success": False
    }
    print("Sending feedback with resolution outcome to /feedback...")
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
    print("Feedback test with resolution outcome completed.")

if __name__ == "__main__":
    test_feedback_resolution()
