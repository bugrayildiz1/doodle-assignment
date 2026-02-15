import pandas as pd

def create_sample():
    df = pd.read_json("support_tickets.json")
    df.sample(100).to_json("support_tickets_sample.json", orient="records")
    print("Sample file created: support_tickets_sample.json")

if __name__ == "__main__":
    create_sample()
