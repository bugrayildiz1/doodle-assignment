from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import networkx as nx
from sentence_transformers import SentenceTransformer, util

router = APIRouter()

def load_tickets_and_graph():
    import re
    TICKETS_PATH = os.getenv("TICKETS_PATH", "support_tickets.json")
    if os.path.exists(TICKETS_PATH):
        tickets = pd.read_json(TICKETS_PATH)
    else:
        tickets = pd.DataFrame()
    G = nx.Graph()
    # Simple regex for error codes (e.g., ERR123, CODE-456, E1001)
    error_code_pattern = re.compile(r"\b([A-Z]{2,10}[-_]?[0-9]{2,6})\b")
    # Simple list of technical terms (expand as needed)
    technical_terms = ["timeout", "API", "DNS", "firewall", "authentication", "token", "endpoint", "connectivity", "latency", "database", "cache", "memory", "CPU", "disk", "network"]
    if not tickets.empty:
        for _, row in tickets.iterrows():
            G.add_node(row['product'], type='product')
            G.add_node(row['category'], type='issue')
            G.add_node(row['resolution'], type='solution')
            G.add_node(row['ticket_id'], type='ticket')
            G.add_edge(row['product'], row['category'])
            G.add_edge(row['category'], row['resolution'])
            G.add_edge(row['ticket_id'], row['category'])
            G.add_edge(row['ticket_id'], row['resolution'])
            # --- Entity linking ---
            text_fields = str(row.get('subject', '')) + ' ' + str(row.get('description', '')) + ' ' + str(row.get('resolution', ''))
            # Error codes
            for code in set(error_code_pattern.findall(text_fields)):
                G.add_node(code, type='error_code')
                G.add_edge(row['ticket_id'], code)
                G.add_edge(row['product'], code)
            # Technical terms
            for term in technical_terms:
                if term.lower() in text_fields.lower():
                    G.add_node(term, type='technical_term')
                    G.add_edge(row['ticket_id'], term)
                    G.add_edge(row['product'], term)
    return tickets, G

def prepare_embeddings(tickets, model):
    if not tickets.empty:
        tickets['text'] = tickets['subject'].fillna('') + ' ' + tickets['description'].fillna('')
        tickets['solution_text'] = tickets['resolution'].fillna('')
        ticket_embeddings = model.encode(tickets['text'].tolist(), convert_to_tensor=True)
        solution_embeddings = model.encode(tickets['solution_text'].tolist(), convert_to_tensor=True)
    else:
        ticket_embeddings = None
        solution_embeddings = None
    return ticket_embeddings, solution_embeddings


class HybridRetrieveInput(BaseModel):
    category: str
    subject: str
    description: str
    product: str


@router.post("/hybrid_retrieve")
def hybrid_retrieve(query: HybridRetrieveInput):
    print("/hybrid_retrieve endpoint called")
    # Load data and build graph on demand
    tickets, G = load_tickets_and_graph()
    if tickets.empty:
        return {"solutions": [], "message": "No tickets loaded."}
    # Prepare model and embeddings on demand
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ticket_embeddings, solution_embeddings = prepare_embeddings(tickets, model)

    # 1. Metadata filtering (category, product)
    filtered = tickets[(tickets["category"] == query.category) & (tickets["product"] == query.product)]
    idxs = filtered.index.tolist()

    # 2. Semantic search on filtered tickets
    query_text = query.subject + ' ' + query.description
    query_emb = model.encode(query_text, convert_to_tensor=True)
    sem_ticket_ids = []
    if idxs:
        filtered_embs = ticket_embeddings[idxs]
        scores = util.pytorch_cos_sim(query_emb, filtered_embs)[0].cpu().numpy()
        top_idx = np.argsort(scores)[::-1][:5]
        sem_ticket_ids = filtered.iloc[top_idx]["ticket_id"].tolist()

    # 3. Keyword matching on all tickets (subject/description)
    keyword_mask = tickets["subject"].str.contains(query.subject, case=False, na=False) | \
                   tickets["description"].str.contains(query.description, case=False, na=False)
    keyword_ticket_ids = tickets[keyword_mask]["ticket_id"].tolist()

    # 4. Combine and deduplicate ticket IDs
    combined_ticket_ids = list(dict.fromkeys(sem_ticket_ids + keyword_ticket_ids))

    # 5. Graph expansion: find related solutions and their ticket info
    solution_info = []
    for tid in combined_ticket_ids:
        if tid in G:
            for neighbor in G.neighbors(tid):
                if G.nodes[neighbor].get('type') == 'solution':
                    # Find the ticket row(s) with this solution
                    ticket_rows = tickets[(tickets['ticket_id'] == tid) & (tickets['resolution'] == neighbor)]
                    for _, row in ticket_rows.iterrows():
                        solution_info.append({
                            'solution': neighbor,
                            'satisfaction_score': row.get('satisfaction_score', 0),
                            'resolution_helpful': row.get('resolution_helpful', False),
                            'resolution_time_hours': row.get('resolution_time_hours', 1e9)
                        })
    # Re-rank: satisfaction_score desc, resolution_helpful true first, resolution_time_hours asc
    solution_info = sorted(
        solution_info,
        key=lambda x: (-x['satisfaction_score'], not x['resolution_helpful'], x['resolution_time_hours'])
    )
    ranked_solutions = [s['solution'] for s in solution_info]

    # 6. Return results
    return {
        "solutions": ranked_solutions,
        "related_tickets": combined_ticket_ids,
        "semantic_ticket_ids": sem_ticket_ids,
        "keyword_ticket_ids": keyword_ticket_ids,
        "solution_ranking_details": solution_info
    }
