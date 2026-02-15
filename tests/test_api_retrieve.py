import requests

# Adjust if running on a different port or host
API_URL = "http://127.0.0.1:8000/retrieve"

def test_retrieve():
    payload = {
        "category": "Technical Issue",
        "subject": "timeout error",
        "description": "sync failing with timeout"
    }
    response = requests.post(API_URL, json=payload)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

if __name__ == "__main__":
    test_retrieve()
