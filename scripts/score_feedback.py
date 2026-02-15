import json
import os

def compute_quality_score(entry):
    # Example scoring logic (customize as needed):
    score = 0
    if entry.get("resolution_success"):
        score += 0.5
    if entry.get("satisfaction_score") is not None:
        score += (entry["satisfaction_score"] / 10)  # 0.1 to 0.5
    if entry.get("feedback_text"):
        score += 0.1  # Bonus for feedback presence
    return min(score, 1.0)  # Cap at 1.0


def main():
    feedback_path = os.getenv("FEEDBACK_PATH", "feedback_log.json")
    if not os.path.exists(feedback_path):
        print(f"No feedback log found at {feedback_path}")
        return
    with open(feedback_path, "r") as f:
        data = json.load(f)
    for entry in data:
        entry["quality_score"] = compute_quality_score(entry)
    # Save with scores
    with open("feedback_log_scored.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Scored {len(data)} feedback entries. Output: feedback_log_scored.json")

if __name__ == "__main__":
    main()
