import requests

API_URL = "http://127.0.0.1:8000/analyze_anomalies"

def test_analyze_anomalies():
    payload = {
        "known_categories": ["Technical Issue", "Feature Request", "Bug Report", "Account Issue"],
        "retrieval_logs": [],
        "window_days": 7
    }
    print("Sending request to /analyze_anomalies...")
    try:
        response = requests.post(API_URL, json=payload, timeout=15)
        print("Status Code:", response.status_code)
        try:
            print("Response JSON:", response.json())
        except Exception:
            print("Response Text:", response.text)
    except Exception as e:
        import traceback
        print("Exception occurred:", e)
        traceback.print_exc()
    print("Request completed.")

if __name__ == "__main__":
    test_analyze_anomalies()
