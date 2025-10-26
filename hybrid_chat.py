# hybrid_chat.py (FINAL VERSION - ALL TASK 3 IMPROVEMENTS)
import json
import asyncio
from typing import List
from collections import Counter
import google.generativeai as genai
from pinecone import Pinecone
from neo4j import AsyncGraphDatabase
import config

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-2.5-pro"  # Using Google's best reasoning model as requested
TOP_K = 5

INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
genai.configure(api_key=config.GOOGLE_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# --- TASK 3: ASYNC IMPROVEMENT ---
# Use Async Driver for Neo4j for non-blocking I/O
driver = AsyncGraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)
# -----------------------------------

INDEX_HOST = "vietnam-travel-google-5308776.svc.aped-4627-b74a.pinecone.io"
index = pc.Index(host=INDEX_HOST)

# --- TASK 3: CACHING IMPROVEMENT ---
embedding_cache = {}
# -----------------------------------

# -----------------------------
# Helper functions (now async)
# -----------------------------
async def embed_text(text: str) -> List[float]:
    """Get embedding for a text string using Google AI (async)."""
    response = await genai.embed_content_async(model=EMBED_MODEL, content=text)
    return response['embedding']

async def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding, with caching (async)."""
    if query_text in embedding_cache:
        print("DEBUG: Using cached embedding.")
        vec = embedding_cache[query_text]
    else:
        print("DEBUG: Generating new embedding.")
        vec = await embed_text(query_text)
        embedding_cache[query_text] = vec

    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        None,
        lambda: index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
    )
    
    print(f"DEBUG: Pinecone top {top_k} results:")
    print(len(res["matches"]))
    return res["matches"]

async def fetch_graph_context(node_ids: List[str]):
    """Fetch neighboring nodes from Neo4j (async)."""
    facts = []
    async with driver.session() as session:
        for nid in node_ids:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type "
                "LIMIT 5"
            )
            recs = await session.run(q, nid=nid)
            async for r in recs:
                facts.append({
                    "source": nid, "rel": r["rel"], "target_id": r["id"],
                    "target_name": r["name"], "labels": r["labels"]
                })
    print(f"DEBUG: Graph facts found: {len(facts)}")
    return facts

# --- TASK 3: SUMMARY FUNCTION ---
def summarize_pinecone_results(matches: List[dict]) -> str:
    """Create a one-sentence summary of the top Pinecone results."""
    if not matches:
        return "No relevant information was found."
    
    tags = Counter()
    cities = Counter()
    for m in matches:
        meta = m.get("metadata", {})
        if "tags" in meta:
            tags.update(t for t in meta["tags"] if t not in ['experience', 'stay'])
        if "city" in meta:
            cities[meta["city"]] += 1

    top_theme = tags.most_common(1)[0][0] if tags else "general"
    top_city = cities.most_common(1)[0][0] if cities else "various locations"

    return f"The most relevant results are related to '{top_theme}' experiences, primarily in {top_city}."
# ---------------------------------

# In hybrid_chat.py, replace the old build_prompt function with this one:

def build_prompt(user_query, pinecone_matches, graph_facts, summary):
    """Build a chat prompt with all context and improved instructions."""
    system_instruction = "You are an expert travel assistant. Your goal is to create a helpful, data-driven travel itinerary based on the user's query and the context provided."

    vec_context = [f"- id: {m['id']}, name: {m['metadata'].get('name', '')}" for m in pinecone_matches]
    graph_context = [f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}" for f in graph_facts]

    # --- START OF PROMPT FIX ---
    # New instructions are more flexible and guide the model based on user intent.
    final_instruction = (
        "Based on the provided data, your task is to answer the user's query.\n\n"
        "INSTRUCTIONS:\n"
        "1. **Analyze the Query**: First, understand what the user is asking for. Are they requesting a full itinerary, a specific recommendation, or general information?\n"
        "2. **Formulate the Answer**: Use the context to directly answer their question. \n"
        "   - If they ask for an **itinerary**, provide a clear, day-by-day plan.\n"
        "   - If they ask a **'where' or 'what' question**, provide a direct recommendation (e.g., 'The best place for X is Y') and explain why.\n"
        "   - If they ask for **information**, summarize the relevant details in a helpful paragraph.\n"
        "3. **Explain Your Reasoning**: Briefly explain *why* you are making your recommendation, referencing the context summary or specific data points.\n"
        "4. **Cite Sources**: When you mention a specific place, cite its `id` in parentheses, like `(attraction_123)`."
    )
    # --- END OF PROMPT FIX ---

    full_prompt = (
        f"{system_instruction}\n\n"
        f"User query: {user_query}\n\n"
        f"## Context Summary\n{summary}\n\n"
        "## Detailed Context\n"
        "Vector Search Results:\n" + "\n".join(vec_context) + "\n\n"
        "Knowledge Graph Connections:\n" + "\n".join(graph_context) + "\n\n"
        f"## Instructions\n{final_instruction}"
    )
    return full_prompt

async def call_chat(prompt):
    """Call Google's Gemini model asynchronously."""
    model = genai.GenerativeModel(CHAT_MODEL)
    response = await model.generate_content_async(prompt)
    return response.text

# -----------------------------
# Main async chat loop
# -----------------------------
async def main_chat_loop():
    print("Hybrid travel assistant (v2 - Fully Improved!). Type 'exit' to quit.")
    while True:
        query = await asyncio.to_thread(input, "\nEnter your travel question: ")
        query = query.strip()

        if not query or query.lower() in ("exit", "quit"):
            break

        pinecone_matches = await pinecone_query(query, top_k=TOP_K)
        if not pinecone_matches:
            print("\n=== Assistant Answer ===\nI couldn't find any relevant travel information for your query.\n=== End ===\n")
            continue
            
        summary = summarize_pinecone_results(pinecone_matches)
        match_ids = [m["id"] for m in pinecone_matches]
        graph_facts = await fetch_graph_context(match_ids)

        prompt = build_prompt(query, pinecone_matches, graph_facts, summary)
        answer = await call_chat(prompt)
        
        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")

if __name__ == "__main__":
    try:
        asyncio.run(main_chat_loop())
    finally:
        asyncio.run(driver.close())