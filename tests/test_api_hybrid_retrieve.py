import requests

API_URL = "http://127.0.0.1:8000/hybrid_retrieve"

def test_hybrid_retrieve():
    payload = {
        "category": "Technical Issue",
        "subject": "timeout error",
        "description": "sync failing with timeout",
        "product": "DataSync Pro"
    }
    print("Sending request to /hybrid_retrieve...")
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
    test_hybrid_retrieve()
