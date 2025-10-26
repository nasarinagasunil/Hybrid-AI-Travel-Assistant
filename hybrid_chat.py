# hybrid_chat.py (MODIFIED FOR GOOGLE AI + TASK 3 IMPROVEMENTS)
import json
from typing import List
import google.generativeai as genai
from pinecone import Pinecone
from neo4j import GraphDatabase
import config

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-2.5-pro"  # Using Google's powerful reasoning model
TOP_K = 5

INDEX_NAME = config.PINECONE_INDEX_NAME  # Should be 'vietnam-travel-google'

# -----------------------------
# Initialize clients
# -----------------------------
# Configure Google client
genai.configure(api_key=config.GOOGLE_API_KEY)

# Configure Pinecone client (modern v3+ syntax)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index using the host
INDEX_HOST = "vietnam-travel-google-5308776.svc.aped-4627-b74a.pinecone.io"
index = pc.Index(host=INDEX_HOST)

# Connect to Neo4j
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# --- TASK 3 IMPROVEMENT: EMBEDDING CACHING ---
# Simple in-memory cache to store embeddings for repeated queries
embedding_cache = {}
# ---------------------------------------------

# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str) -> List[float]:
    """Get embedding for a text string using Google AI."""
    response = genai.embed_content(model=EMBED_MODEL, content=text)
    return response['embedding']

def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding, with caching."""
    # --- TASK 3 IMPROVEMENT: EMBEDDING CACHING ---
    if query_text in embedding_cache:
        print("DEBUG: Using cached embedding.")
        vec = embedding_cache[query_text]
    else:
        print("DEBUG: Generating new embedding.")
        vec = embed_text(query_text)
        embedding_cache[query_text] = vec
    # ---------------------------------------------

    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    print(f"DEBUG: Pinecone top {top_k} results:")
    print(len(res["matches"]))
    return res["matches"]

def fetch_graph_context(node_ids: List[str]):
    """Fetch neighboring nodes from Neo4j."""
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "LIMIT 10"
            )
            recs = session.run(q, nid=nid)
            for r in recs:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:400],
                    "labels": r["labels"]
                })
    print("DEBUG: Graph facts:")
    print(len(facts))
    return facts

def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a chat prompt combining vector DB matches and graph facts."""
    system_instruction = (
        "You are an expert travel assistant. Your goal is to create a helpful, "
        "data-driven travel itinerary based on the user's query and the context provided."
    )

    vec_context = []
    for m in pinecone_matches:
        meta = m["metadata"]
        score = m.get("score", None)
        snippet = f"- id: {m['id']}, name: {meta.get('name','')}, type: {meta.get('type','')}, score: {score}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}"
        for f in graph_facts
    ]

    # --- TASK 3 IMPROVEMENT: CHAIN-OF-THOUGHT PROMPT ---
    final_instruction = (
        "Based on the provided data, perform the following steps:\n"
        "1.  **Reasoning**: First, in a paragraph, explain your reasoning. Analyze the user's query and the retrieved context (both semantic and graph facts) to decide on the best travel plan. Mention which cities or themes are most relevant and why.\n"
        "2.  **Itinerary**: Second, generate a concise, day-by-day travel itinerary based on your reasoning. Format the output clearly using markdown. Cite specific node ids (e.g., `attraction_123`) for attractions or activities where appropriate."
    )

    full_prompt = (
        f"{system_instruction}\n\n"
        f"User query: {user_query}\n\n"
        "## Context from Vector Database\n" + "\n".join(vec_context) + "\n\n"
        "## Context from Knowledge Graph\n" + "\n".join(graph_context) + "\n\n"
        f"## Instructions\n{final_instruction}"
    )
    # ---------------------------------------------
    return full_prompt

def call_chat(prompt):
    """Call Google's Gemini model."""
    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(prompt)
    return response.text

# -----------------------------
# Interactive chat
# -----------------------------
def interactive_chat():
    print("Hybrid travel assistant (Google AI Edition - Improved!). Type 'exit' to quit.")
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break

        matches = pinecone_query(query, top_k=TOP_K)
        match_ids = [m["id"] for m in matches]
        graph_facts = fetch_graph_context(match_ids)
        prompt = build_prompt(query, matches, graph_facts)
        answer = call_chat(prompt)
        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")

if __name__ == "__main__":
    interactive_chat()